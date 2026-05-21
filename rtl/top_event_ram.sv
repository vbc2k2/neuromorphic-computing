`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top_event_ram.sv - Area-oriented event-driven SNN top level
//
// Description:
//   V3 prototype that keeps neuron membrane/accumulator state in indexed arrays
//   and uses one shared neuron update datapath. This avoids instantiating one
//   physical neuron_core per logical neuron. It is intended for area comparison
//   against the spatial event top, not as a final high-throughput design.
//
//   Operation per timestep:
//     1. While idle, external source spikes are queued.
//     2. On start_step, queued source spikes are routed through CSR fan-out.
//        Each delivery accumulates weight_sum[dst].
//     3. The design scans all logical neurons once, applies leak/integrate/
//        threshold, emits one-cycle spike pulses, and queues fired neurons for
//        the next timestep.
//-----------------------------------------------------------------------------

module top_event_ram #(
    parameter int NUM_NEURONS    = 139,
    parameter int WEIGHT_WIDTH   = 8,
    parameter int MEMBRANE_WIDTH = 16,
    parameter int THRESHOLD      = 210,
    parameter int LEAK           = 1,
    parameter int NUM_SYN        = 4810,
    parameter int FIFO_DEPTH     = 4096,
    parameter     CSR_DST_FILE    = "csr_dst.mem",
    parameter     CSR_WEIGHT_FILE = "csr_weight.mem",
    parameter     CSR_START_FILE  = "csr_start.mem",
    parameter     CSR_COUNT_FILE  = "csr_count.mem"
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic [$clog2(NUM_NEURONS)-1:0] ext_spike_id,
    input  logic                        ext_spike_valid,

    input  logic                        start_step,
    output logic                        step_busy,

    output logic [NUM_NEURONS-1:0]      neuron_spikes,

    output logic [31:0]                 cycle_count,
    output logic [31:0]                 active_cycles,
    output logic [31:0]                 total_spikes_fired,
    output logic [31:0]                 router_events,
    output logic [31:0]                 router_deliveries
);

    localparam int ID_WIDTH         = $clog2(NUM_NEURONS);
    localparam int IDX_WIDTH        = $clog2(NUM_SYN + 1);
    localparam int FIFO_ADDR_WIDTH  = $clog2(FIFO_DEPTH);
    localparam int FIFO_COUNT_WIDTH = $clog2(FIFO_DEPTH) + 1;

    // =========================================================================
    // CSR Synapse Connectivity Store
    // =========================================================================
    logic [ID_WIDTH-1:0]            csr_ptr_src;
    logic [IDX_WIDTH-1:0]           csr_ptr_base;
    logic [IDX_WIDTH-1:0]           csr_ptr_len;
    logic [IDX_WIDTH-1:0]           csr_syn_index;
    logic                           csr_syn_rd_en;
    logic [ID_WIDTH-1:0]            csr_syn_dst;
    logic signed [WEIGHT_WIDTH-1:0] csr_syn_weight;
    logic                           csr_syn_valid;

    synapse_csr #(
        .NUM_NEURONS     (NUM_NEURONS),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH),
        .NUM_SYN         (NUM_SYN),
        .CSR_DST_FILE    (CSR_DST_FILE),
        .CSR_WEIGHT_FILE (CSR_WEIGHT_FILE),
        .CSR_START_FILE  (CSR_START_FILE),
        .CSR_COUNT_FILE  (CSR_COUNT_FILE)
    ) u_synapse_csr (
        .clk        (clk),
        .ptr_src_id (csr_ptr_src),
        .ptr_base   (csr_ptr_base),
        .ptr_len    (csr_ptr_len),
        .syn_index  (csr_syn_index),
        .syn_rd_en  (csr_syn_rd_en),
        .syn_dst    (csr_syn_dst),
        .syn_weight (csr_syn_weight),
        .syn_valid  (csr_syn_valid)
    );

    // =========================================================================
    // Source event FIFO
    // =========================================================================
    logic [ID_WIDTH-1:0]              fifo_mem [0:FIFO_DEPTH-1];
    logic [FIFO_ADDR_WIDTH-1:0]       fifo_wr_ptr;
    logic [FIFO_ADDR_WIDTH-1:0]       fifo_rd_ptr;
    logic [FIFO_COUNT_WIDTH-1:0]      fifo_count;
    logic                             fifo_push;
    logic                             fifo_pop;
    logic [ID_WIDTH-1:0]              fifo_push_id;
    logic                             fifo_full;
    logic                             fifo_empty;

    assign fifo_full  = (fifo_count == FIFO_COUNT_WIDTH'(FIFO_DEPTH));
    assign fifo_empty = (fifo_count == '0);

    // =========================================================================
    // Shared neuron state
    // =========================================================================
    logic signed [MEMBRANE_WIDTH-1:0] membrane   [0:NUM_NEURONS-1];
    logic signed [MEMBRANE_WIDTH-1:0] weight_sum [0:NUM_NEURONS-1];

    initial begin
        for (int i = 0; i < NUM_NEURONS; i++) begin
            membrane[i]   = '0;
            weight_sum[i] = '0;
        end
    end

    function automatic logic signed [MEMBRANE_WIDTH-1:0]
        sext_weight(input logic signed [WEIGHT_WIDTH-1:0] value);
        sext_weight = {{(MEMBRANE_WIDTH-WEIGHT_WIDTH){value[WEIGHT_WIDTH-1]}}, value};
    endfunction

    function automatic logic signed [MEMBRANE_WIDTH-1:0]
        leak_value(input logic signed [MEMBRANE_WIDTH-1:0] value);
        begin
            if (value > MEMBRANE_WIDTH'(signed'(LEAK)))
                leak_value = value - MEMBRANE_WIDTH'(signed'(LEAK));
            else if (value > 0)
                leak_value = '0;
            else
                leak_value = value;
        end
    endfunction

    // =========================================================================
    // Controller
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_DEQUEUE,
        S_CHECK,
        S_STREAM,
        S_SCAN,
        S_DONE
    } state_t;

    state_t state, state_next;
    logic [ID_WIDTH-1:0]  current_src;
    logic [IDX_WIDTH-1:0] issue_idx;
    logic [IDX_WIDTH-1:0] deliver_idx;
    logic [ID_WIDTH-1:0]  scan_idx;

    logic issue_valid;
    logic delivery_valid;
    logic last_delivery;
    logic scan_last;
    logic signed [MEMBRANE_WIDTH-1:0] scanned_leaked;
    logic signed [MEMBRANE_WIDTH-1:0] scanned_integrated;
    logic scanned_fired;

    assign step_busy      = (state != S_IDLE);
    assign csr_ptr_src    = current_src;
    assign csr_syn_index  = csr_ptr_base + issue_idx;
    assign issue_valid    = (state == S_STREAM) && (issue_idx < csr_ptr_len);
    assign csr_syn_rd_en  = issue_valid;
    assign delivery_valid = (state == S_STREAM) && csr_syn_valid;
    assign last_delivery  = delivery_valid && (deliver_idx + IDX_WIDTH'(1) == csr_ptr_len);
    assign scan_last      = (scan_idx == ID_WIDTH'(NUM_NEURONS - 1));

    always_comb begin
        scanned_leaked     = leak_value(membrane[scan_idx]);
        scanned_integrated = scanned_leaked + weight_sum[scan_idx];
        scanned_fired      = (scanned_integrated >= MEMBRANE_WIDTH'(signed'(THRESHOLD)));
    end

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE: begin
                if (start_step)
                    state_next = fifo_empty ? S_SCAN : S_DEQUEUE;
            end
            S_DEQUEUE: state_next = S_CHECK;
            S_CHECK:   state_next = (csr_ptr_len == 0) ? (fifo_empty ? S_SCAN : S_DEQUEUE) : S_STREAM;
            S_STREAM: begin
                if (last_delivery)
                    state_next = fifo_empty ? S_SCAN : S_DEQUEUE;
            end
            S_SCAN:    if (scan_last) state_next = S_DONE;
            S_DONE:    state_next = S_IDLE;
            default:   state_next = S_IDLE;
        endcase
    end

    always_comb begin
        fifo_push    = 1'b0;
        fifo_push_id = ext_spike_id;
        fifo_pop     = 1'b0;

        if (state == S_IDLE && ext_spike_valid && !fifo_full) begin
            fifo_push    = 1'b1;
            fifo_push_id = ext_spike_id;
        end else if (state == S_SCAN && scanned_fired && !fifo_full) begin
            fifo_push    = 1'b1;
            fifo_push_id = scan_idx;
        end

        if (state == S_DEQUEUE)
            fifo_pop = 1'b1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            fifo_wr_ptr        <= '0;
            fifo_rd_ptr        <= '0;
            fifo_count         <= '0;
            current_src        <= '0;
            issue_idx          <= '0;
            deliver_idx        <= '0;
            scan_idx           <= '0;
            neuron_spikes      <= '0;
            cycle_count        <= '0;
            active_cycles      <= '0;
            total_spikes_fired <= '0;
            router_events      <= '0;
            router_deliveries  <= '0;
        end else begin
            state         <= state_next;
            cycle_count   <= cycle_count + 1;
            neuron_spikes <= '0;

            if (state != S_IDLE)
                active_cycles <= active_cycles + 1;

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1;
                2'b01: fifo_count <= fifo_count - 1;
                default: ;
            endcase

            if (fifo_push) begin
                fifo_mem[fifo_wr_ptr] <= fifo_push_id;
                fifo_wr_ptr <= fifo_wr_ptr + 1;
            end

            if (fifo_pop)
                fifo_rd_ptr <= fifo_rd_ptr + 1;

            case (state)
                S_IDLE: begin
                    if (start_step)
                        scan_idx <= '0;
                end

                S_DEQUEUE: begin
                    current_src   <= fifo_mem[fifo_rd_ptr];
                    issue_idx     <= '0;
                    deliver_idx   <= '0;
                    router_events <= router_events + 1;
                end

                S_STREAM: begin
                    if (issue_valid)
                        issue_idx <= issue_idx + 1;
                    if (delivery_valid) begin
                        weight_sum[csr_syn_dst] <= weight_sum[csr_syn_dst] + sext_weight(csr_syn_weight);
                        deliver_idx             <= deliver_idx + 1;
                        router_deliveries       <= router_deliveries + 1;
                    end
                end

                S_SCAN: begin
                    weight_sum[scan_idx] <= '0;
                    if (scanned_fired) begin
                        membrane[scan_idx] <= '0;
                        neuron_spikes[scan_idx] <= 1'b1;
                        total_spikes_fired <= total_spikes_fired + 1;
                    end else begin
                        membrane[scan_idx] <= scanned_integrated;
                    end
                    if (!scan_last)
                        scan_idx <= scan_idx + 1;
                end

                default: ;
            endcase
        end
    end

endmodule
