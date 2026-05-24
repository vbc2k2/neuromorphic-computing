`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top_event_ram.sv - Area-oriented event-driven SNN top level
//
// Description:
//   Event SNN top level that time-multiplexes neuron state through indexed
//   memories instead of instantiating one neuron_core per logical neuron.
//   The large state arrays are written in a synchronous RAM style so open
//   synthesis can map them away from flip-flop banks.
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
    // BRAM-style source event FIFO
    // =========================================================================
    (* ram_style = "block" *) logic [ID_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];

    logic [FIFO_ADDR_WIDTH-1:0]  fifo_wr_ptr;
    logic [FIFO_ADDR_WIDTH-1:0]  fifo_rd_ptr;
    logic [FIFO_COUNT_WIDTH-1:0] fifo_count;
    logic [ID_WIDTH-1:0]         fifo_rd_data;
    logic                        fifo_push;
    logic                        fifo_pop;
    logic [ID_WIDTH-1:0]         fifo_push_id;
    logic                        fifo_full;
    logic                        fifo_empty;

    assign fifo_full  = (fifo_count == FIFO_COUNT_WIDTH'(FIFO_DEPTH));
    assign fifo_empty = (fifo_count == '0);

    always_ff @(posedge clk) begin
        if (fifo_pop)
            fifo_rd_data <= fifo_mem[fifo_rd_ptr];
        if (fifo_push)
            fifo_mem[fifo_wr_ptr] <= fifo_push_id;
    end

    // =========================================================================
    // BRAM-style shared neuron state
    // =========================================================================
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] membrane   [0:NUM_NEURONS-1];
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] weight_sum [0:NUM_NEURONS-1];

    logic                        mem_rd_en;
    logic                        mem_wr_en;
    logic [ID_WIDTH-1:0]         mem_rd_addr;
    logic [ID_WIDTH-1:0]         mem_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] mem_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] mem_wr_data;

    logic                        ws_rd_en;
    logic                        ws_wr_en;
    logic [ID_WIDTH-1:0]         ws_rd_addr;
    logic [ID_WIDTH-1:0]         ws_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] ws_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] ws_wr_data;

    always_ff @(posedge clk) begin
        if (mem_rd_en)
            mem_rd_data <= membrane[mem_rd_addr];
        if (mem_wr_en)
            membrane[mem_wr_addr] <= mem_wr_data;
    end

    always_ff @(posedge clk) begin
        if (ws_rd_en)
            ws_rd_data <= weight_sum[ws_rd_addr];
        if (ws_wr_en)
            weight_sum[ws_wr_addr] <= ws_wr_data;
    end

    function automatic logic signed [MEMBRANE_WIDTH-1:0]
        sext_weight(input logic signed [WEIGHT_WIDTH-1:0] value);
        sext_weight = {{(MEMBRANE_WIDTH-WEIGHT_WIDTH){value[WEIGHT_WIDTH-1]}}, value};
    endfunction

    function automatic logic signed [MEMBRANE_WIDTH-1:0]
        leak_value(input logic signed [MEMBRANE_WIDTH-1:0] value);
        logic signed [MEMBRANE_WIDTH-1:0] leak_amount;
        begin
            leak_amount = MEMBRANE_WIDTH'(signed'(LEAK));
            if (value > leak_amount)
                leak_value = value - leak_amount;
            else if (value > MEMBRANE_WIDTH'(signed'(0)))
                leak_value = MEMBRANE_WIDTH'(signed'(0));
            else
                leak_value = value;
        end
    endfunction

    // =========================================================================
    // Controller
    // =========================================================================
    typedef enum logic [3:0] {
        S_CLEAR,
        S_IDLE,
        S_FIFO_READ,
        S_SRC_LATCH,
        S_CHECK,
        S_CSR_ISSUE,
        S_DELIV_READ,
        S_DELIV_WRITE,
        S_SCAN_READ,
        S_SCAN_WRITE,
        S_DONE
    } state_t;

    state_t state, state_next;

    logic [ID_WIDTH-1:0]  clear_idx;
    logic [ID_WIDTH-1:0]  current_src;
    logic [IDX_WIDTH-1:0] issue_idx;
    logic [ID_WIDTH-1:0]  pending_dst;
    logic signed [WEIGHT_WIDTH-1:0] pending_weight;
    logic [ID_WIDTH-1:0]  scan_idx;

    logic clear_last;
    logic scan_last;
    logic signed [MEMBRANE_WIDTH-1:0] scanned_leaked;
    logic signed [MEMBRANE_WIDTH-1:0] scanned_integrated;
    logic scanned_fired;

    assign step_busy      = (state != S_IDLE);
    assign csr_ptr_src    = current_src;
    assign csr_syn_index  = csr_ptr_base + issue_idx;
    assign csr_syn_rd_en  = (state == S_CSR_ISSUE);
    assign clear_last     = (clear_idx == ID_WIDTH'(NUM_NEURONS - 1));
    assign scan_last      = (scan_idx == ID_WIDTH'(NUM_NEURONS - 1));

    always_comb begin
        scanned_leaked     = leak_value(mem_rd_data);
        scanned_integrated = scanned_leaked + ws_rd_data;
        scanned_fired      = (scanned_integrated >= MEMBRANE_WIDTH'(signed'(THRESHOLD)));
    end

    always_comb begin
        state_next = state;
        case (state)
            S_CLEAR:      state_next = clear_last ? S_IDLE : S_CLEAR;
            S_IDLE:       if (start_step) state_next = fifo_empty ? S_SCAN_READ : S_FIFO_READ;
            S_FIFO_READ:  state_next = S_SRC_LATCH;
            S_SRC_LATCH:  state_next = S_CHECK;
            S_CHECK:      state_next = (csr_ptr_len == '0) ?
                                      (fifo_empty ? S_SCAN_READ : S_FIFO_READ) : S_CSR_ISSUE;
            S_CSR_ISSUE:  state_next = S_DELIV_READ;
            S_DELIV_READ: state_next = csr_syn_valid ? S_DELIV_WRITE : S_DELIV_READ;
            S_DELIV_WRITE: state_next = (issue_idx < csr_ptr_len) ?
                                       S_CSR_ISSUE :
                                       (fifo_empty ? S_SCAN_READ : S_FIFO_READ);
            S_SCAN_READ:  state_next = S_SCAN_WRITE;
            S_SCAN_WRITE: state_next = scan_last ? S_DONE : S_SCAN_READ;
            S_DONE:       state_next = S_IDLE;
            default:      state_next = S_CLEAR;
        endcase
    end

    always_comb begin
        fifo_push    = 1'b0;
        fifo_push_id = ext_spike_id;
        fifo_pop     = 1'b0;

        if (state == S_IDLE && ext_spike_valid && !fifo_full) begin
            fifo_push    = 1'b1;
            fifo_push_id = ext_spike_id;
        end else if (state == S_SCAN_WRITE && scanned_fired && !fifo_full) begin
            fifo_push    = 1'b1;
            fifo_push_id = scan_idx;
        end

        if (state == S_FIFO_READ)
            fifo_pop = 1'b1;
    end

    always_comb begin
        mem_rd_en   = 1'b0;
        mem_rd_addr = scan_idx;
        mem_wr_en   = 1'b0;
        mem_wr_addr = scan_idx;
        mem_wr_data = scanned_fired ? '0 : scanned_integrated;

        ws_rd_en    = 1'b0;
        ws_rd_addr  = scan_idx;
        ws_wr_en    = 1'b0;
        ws_wr_addr  = scan_idx;
        ws_wr_data  = '0;

        case (state)
            S_CLEAR: begin
                mem_wr_en   = 1'b1;
                mem_wr_addr = clear_idx;
                mem_wr_data = '0;
                ws_wr_en    = 1'b1;
                ws_wr_addr  = clear_idx;
                ws_wr_data  = '0;
            end

            S_DELIV_READ: begin
                ws_rd_en   = csr_syn_valid;
                ws_rd_addr = csr_syn_dst;
            end

            S_DELIV_WRITE: begin
                ws_wr_en   = 1'b1;
                ws_wr_addr = pending_dst;
                ws_wr_data = ws_rd_data + sext_weight(pending_weight);
            end

            S_SCAN_READ: begin
                mem_rd_en   = 1'b1;
                mem_rd_addr = scan_idx;
                ws_rd_en    = 1'b1;
                ws_rd_addr  = scan_idx;
            end

            S_SCAN_WRITE: begin
                mem_wr_en   = 1'b1;
                mem_wr_addr = scan_idx;
                mem_wr_data = scanned_fired ? '0 : scanned_integrated;
                ws_wr_en    = 1'b1;
                ws_wr_addr  = scan_idx;
                ws_wr_data  = '0;
            end

            default: ;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_CLEAR;
            fifo_wr_ptr        <= '0;
            fifo_rd_ptr        <= '0;
            fifo_count         <= '0;
            clear_idx          <= '0;
            current_src        <= '0;
            issue_idx          <= '0;
            pending_dst        <= '0;
            pending_weight     <= '0;
            scan_idx           <= '0;
            neuron_spikes      <= '0;
            cycle_count        <= '0;
            active_cycles      <= '0;
            total_spikes_fired <= '0;
            router_events      <= '0;
            router_deliveries  <= '0;
        end else begin
            state         <= state_next;
            cycle_count   <= cycle_count + 32'd1;
            neuron_spikes <= '0;

            if (state != S_IDLE && state != S_CLEAR)
                active_cycles <= active_cycles + 32'd1;

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + FIFO_COUNT_WIDTH'(1);
                2'b01: fifo_count <= fifo_count - FIFO_COUNT_WIDTH'(1);
                default: ;
            endcase

            if (fifo_push)
                fifo_wr_ptr <= fifo_wr_ptr + FIFO_ADDR_WIDTH'(1);

            if (fifo_pop)
                fifo_rd_ptr <= fifo_rd_ptr + FIFO_ADDR_WIDTH'(1);

            case (state)
                S_CLEAR: begin
                    if (!clear_last)
                        clear_idx <= clear_idx + ID_WIDTH'(1);
                end

                S_IDLE: begin
                    if (start_step)
                        scan_idx <= '0;
                end

                S_SRC_LATCH: begin
                    current_src   <= fifo_rd_data;
                    issue_idx     <= '0;
                    router_events <= router_events + 32'd1;
                end

                S_CSR_ISSUE: begin
                    issue_idx <= issue_idx + IDX_WIDTH'(1);
                end

                S_DELIV_READ: begin
                    if (csr_syn_valid) begin
                        pending_dst    <= csr_syn_dst;
                        pending_weight <= csr_syn_weight;
                    end
                end

                S_DELIV_WRITE: begin
                    router_deliveries <= router_deliveries + 32'd1;
                end

                S_SCAN_WRITE: begin
                    if (scanned_fired) begin
                        neuron_spikes[scan_idx] <= 1'b1;
                        total_spikes_fired <= total_spikes_fired + 32'd1;
                    end
                    if (!scan_last)
                        scan_idx <= scan_idx + ID_WIDTH'(1);
                end

                default: ;
            endcase
        end
    end

endmodule
