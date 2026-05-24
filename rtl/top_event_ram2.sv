`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top_event_ram2.sv - Two-lane RAM-state event-driven SNN top level
//
// Description:
//   Area-oriented event SNN with two banked neuron-state scan lanes. Synapse
//   delivery remains one event at a time, while timestep scan/threshold work is
//   performed over even/odd neuron banks in parallel.
//-----------------------------------------------------------------------------

module top_event_ram2 #(
    parameter int NUM_NEURONS    = 139,
    parameter int NUM_OUTPUT     = 10,
    parameter int ID_OUTPUT_BASE = 129,
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

    input  logic [$clog2(NUM_OUTPUT)-1:0] score_read_idx,
    input  logic                        score_read_en,
    output logic                        score_read_valid,
    output logic signed [MEMBRANE_WIDTH-1:0] score_read_data,

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
    localparam int BANK0_DEPTH      = (NUM_NEURONS + 1) / 2;
    localparam int BANK1_DEPTH      = NUM_NEURONS / 2;
    localparam int BANK_ADDR_WIDTH  = (BANK0_DEPTH <= 1) ? 1 : $clog2(BANK0_DEPTH);
    localparam int ID_EXT_WIDTH     = ID_WIDTH + 1;
    localparam logic [ID_EXT_WIDTH-1:0] NUM_NEURONS_EXT = ID_EXT_WIDTH'(NUM_NEURONS);

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
    // Banked BRAM-style neuron state
    // =========================================================================
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] membrane0   [0:BANK0_DEPTH-1];
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] membrane1   [0:BANK1_DEPTH-1];
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] weight_sum0 [0:BANK0_DEPTH-1];
    (* ram_style = "block" *) logic signed [MEMBRANE_WIDTH-1:0] weight_sum1 [0:BANK1_DEPTH-1];

    logic mem0_rd_en;
    logic mem0_wr_en;
    logic [BANK_ADDR_WIDTH-1:0] mem0_rd_addr;
    logic [BANK_ADDR_WIDTH-1:0] mem0_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] mem0_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] mem0_wr_data;

    logic mem1_rd_en;
    logic mem1_wr_en;
    logic [BANK_ADDR_WIDTH-1:0] mem1_rd_addr;
    logic [BANK_ADDR_WIDTH-1:0] mem1_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] mem1_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] mem1_wr_data;

    logic ws0_rd_en;
    logic ws0_wr_en;
    logic [BANK_ADDR_WIDTH-1:0] ws0_rd_addr;
    logic [BANK_ADDR_WIDTH-1:0] ws0_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] ws0_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] ws0_wr_data;

    logic ws1_rd_en;
    logic ws1_wr_en;
    logic [BANK_ADDR_WIDTH-1:0] ws1_rd_addr;
    logic [BANK_ADDR_WIDTH-1:0] ws1_wr_addr;
    logic signed [MEMBRANE_WIDTH-1:0] ws1_rd_data;
    logic signed [MEMBRANE_WIDTH-1:0] ws1_wr_data;

    always_ff @(posedge clk) begin
        if (mem0_rd_en)
            mem0_rd_data <= membrane0[mem0_rd_addr];
        if (mem0_wr_en)
            membrane0[mem0_wr_addr] <= mem0_wr_data;
    end

    always_ff @(posedge clk) begin
        if (mem1_rd_en)
            mem1_rd_data <= membrane1[mem1_rd_addr];
        if (mem1_wr_en)
            membrane1[mem1_wr_addr] <= mem1_wr_data;
    end

    always_ff @(posedge clk) begin
        if (ws0_rd_en)
            ws0_rd_data <= weight_sum0[ws0_rd_addr];
        if (ws0_wr_en)
            weight_sum0[ws0_wr_addr] <= ws0_wr_data;
    end

    always_ff @(posedge clk) begin
        if (ws1_rd_en)
            ws1_rd_data <= weight_sum1[ws1_rd_addr];
        if (ws1_wr_en)
            weight_sum1[ws1_wr_addr] <= ws1_wr_data;
    end

    function automatic logic [BANK_ADDR_WIDTH-1:0]
        bank_addr(input logic [ID_WIDTH-1:0] neuron_id);
        bank_addr = BANK_ADDR_WIDTH'(neuron_id >> 1);
    endfunction

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
        S_SCAN_PUSH_ODD,
        S_DONE
    } state_t;

    state_t state, state_next;

    logic [BANK_ADDR_WIDTH-1:0] clear_idx;
    logic [ID_WIDTH-1:0]        current_src;
    logic [IDX_WIDTH-1:0]       issue_idx;
    logic [ID_WIDTH-1:0]        pending_dst;
    logic signed [WEIGHT_WIDTH-1:0] pending_weight;
    logic [BANK_ADDR_WIDTH-1:0] scan_idx;

    logic [ID_WIDTH-1:0] scan_even_id;
    logic [ID_WIDTH-1:0] scan_odd_id;
    logic [ID_EXT_WIDTH-1:0] scan_odd_id_ext;
    logic clear_odd_valid;
    logic scan_odd_valid;
    logic clear_last;
    logic scan_last;

    logic signed [MEMBRANE_WIDTH-1:0] even_leaked;
    logic signed [MEMBRANE_WIDTH-1:0] even_integrated;
    logic signed [MEMBRANE_WIDTH-1:0] odd_leaked;
    logic signed [MEMBRANE_WIDTH-1:0] odd_integrated;
    logic even_fired;
    logic odd_fired;
    logic scan_even_push;
    logic scan_odd_push;
    logic scan_both_fire;
    logic [1:0] scan_fire_count;
    logic score_read_pending;
    logic score_read_bank;
    logic [ID_WIDTH-1:0] score_read_id;

    assign score_read_id = ID_WIDTH'(ID_OUTPUT_BASE) + ID_WIDTH'(score_read_idx);

    assign step_busy      = (state != S_IDLE);
    assign csr_ptr_src    = current_src;
    assign csr_syn_index  = csr_ptr_base + issue_idx;
    assign csr_syn_rd_en  = (state == S_CSR_ISSUE);

    assign scan_even_id   = ID_WIDTH'(scan_idx) << 1;
    assign scan_odd_id    = (ID_WIDTH'(scan_idx) << 1) + ID_WIDTH'(1);
    assign scan_odd_id_ext = {1'b0, scan_odd_id};
    assign clear_odd_valid = (clear_idx < BANK_ADDR_WIDTH'(BANK1_DEPTH));
    assign scan_odd_valid = (scan_odd_id_ext < NUM_NEURONS_EXT);
    assign clear_last     = (clear_idx == BANK_ADDR_WIDTH'(BANK0_DEPTH - 1));
    assign scan_last      = (scan_idx == BANK_ADDR_WIDTH'(BANK0_DEPTH - 1));

    function automatic logic
        is_output_id(input logic [ID_WIDTH-1:0] neuron_id);
        logic [ID_EXT_WIDTH-1:0] id_ext;
        logic [ID_EXT_WIDTH-1:0] out_base_ext;
        logic [ID_EXT_WIDTH-1:0] out_end_ext;
        begin
            id_ext       = {1'b0, neuron_id};
            out_base_ext = ID_EXT_WIDTH'(ID_OUTPUT_BASE);
            out_end_ext  = ID_EXT_WIDTH'(ID_OUTPUT_BASE + NUM_OUTPUT);
            is_output_id = (id_ext >= out_base_ext) && (id_ext < out_end_ext);
        end
    endfunction

    always_comb begin
        even_leaked      = leak_value(mem0_rd_data);
        even_integrated  = even_leaked + ws0_rd_data;
        even_fired       = (even_integrated >= MEMBRANE_WIDTH'(signed'(THRESHOLD)));

        odd_leaked       = leak_value(mem1_rd_data);
        odd_integrated   = odd_leaked + ws1_rd_data;
        odd_fired        = scan_odd_valid &&
                           (odd_integrated >= MEMBRANE_WIDTH'(signed'(THRESHOLD)));

        scan_even_push   = even_fired && !is_output_id(scan_even_id);
        scan_odd_push    = scan_odd_valid && odd_fired && !is_output_id(scan_odd_id);
        scan_both_fire   = scan_even_push && scan_odd_push;
        scan_fire_count  = {1'b0, scan_even_push} + {1'b0, scan_odd_push};
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
            S_SCAN_WRITE: begin
                if (scan_both_fire)
                    state_next = S_SCAN_PUSH_ODD;
                else
                    state_next = scan_last ? S_DONE : S_SCAN_READ;
            end
            S_SCAN_PUSH_ODD: state_next = scan_last ? S_DONE : S_SCAN_READ;
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
        end else if (state == S_SCAN_WRITE && !fifo_full) begin
            if (scan_even_push) begin
                fifo_push    = 1'b1;
                fifo_push_id = scan_even_id;
            end else if (scan_odd_push) begin
                fifo_push    = 1'b1;
                fifo_push_id = scan_odd_id;
            end
        end else if (state == S_SCAN_PUSH_ODD && !fifo_full) begin
            fifo_push    = 1'b1;
            fifo_push_id = scan_odd_id;
        end

        if (state == S_FIFO_READ)
            fifo_pop = 1'b1;
    end

    always_comb begin
        mem0_rd_en   = 1'b0;
        mem0_rd_addr = scan_idx;
        mem0_wr_en   = 1'b0;
        mem0_wr_addr = scan_idx;
        mem0_wr_data = even_fired ? '0 : even_integrated;

        mem1_rd_en   = 1'b0;
        mem1_rd_addr = scan_idx;
        mem1_wr_en   = 1'b0;
        mem1_wr_addr = scan_idx;
        mem1_wr_data = odd_fired ? '0 : odd_integrated;

        ws0_rd_en    = 1'b0;
        ws0_rd_addr  = scan_idx;
        ws0_wr_en    = 1'b0;
        ws0_wr_addr  = scan_idx;
        ws0_wr_data  = '0;

        ws1_rd_en    = 1'b0;
        ws1_rd_addr  = scan_idx;
        ws1_wr_en    = 1'b0;
        ws1_wr_addr  = scan_idx;
        ws1_wr_data  = '0;

        case (state)
            S_IDLE: begin
                mem0_rd_en   = score_read_en && !score_read_id[0];
                mem0_rd_addr = bank_addr(score_read_id);
                mem1_rd_en   = score_read_en && score_read_id[0];
                mem1_rd_addr = bank_addr(score_read_id);
            end

            S_CLEAR: begin
                mem0_wr_en   = 1'b1;
                mem0_wr_addr = clear_idx;
                mem0_wr_data = '0;
                ws0_wr_en    = 1'b1;
                ws0_wr_addr  = clear_idx;
                ws0_wr_data  = '0;

                mem1_wr_en   = clear_odd_valid;
                mem1_wr_addr = clear_idx;
                mem1_wr_data = '0;
                ws1_wr_en    = clear_odd_valid;
                ws1_wr_addr  = clear_idx;
                ws1_wr_data  = '0;
            end

            S_DELIV_READ: begin
                ws0_rd_en   = csr_syn_valid && !csr_syn_dst[0];
                ws0_rd_addr = bank_addr(csr_syn_dst);
                ws1_rd_en   = csr_syn_valid && csr_syn_dst[0];
                ws1_rd_addr = bank_addr(csr_syn_dst);
            end

            S_DELIV_WRITE: begin
                if (pending_dst[0]) begin
                    ws1_wr_en   = 1'b1;
                    ws1_wr_addr = bank_addr(pending_dst);
                    ws1_wr_data = ws1_rd_data + sext_weight(pending_weight);
                end else begin
                    ws0_wr_en   = 1'b1;
                    ws0_wr_addr = bank_addr(pending_dst);
                    ws0_wr_data = ws0_rd_data + sext_weight(pending_weight);
                end
            end

            S_SCAN_READ: begin
                mem0_rd_en   = 1'b1;
                mem0_rd_addr = scan_idx;
                ws0_rd_en    = 1'b1;
                ws0_rd_addr  = scan_idx;

                mem1_rd_en   = scan_odd_valid;
                mem1_rd_addr = scan_idx;
                ws1_rd_en    = scan_odd_valid;
                ws1_rd_addr  = scan_idx;
            end

            S_SCAN_WRITE: begin
                mem0_wr_en   = 1'b1;
                mem0_wr_addr = scan_idx;
                mem0_wr_data = (even_fired && !is_output_id(scan_even_id)) ? '0 : even_integrated;
                ws0_wr_en    = 1'b1;
                ws0_wr_addr  = scan_idx;
                ws0_wr_data  = '0;

                mem1_wr_en   = scan_odd_valid;
                mem1_wr_addr = scan_idx;
                mem1_wr_data = (odd_fired && !is_output_id(scan_odd_id)) ? '0 : odd_integrated;
                ws1_wr_en    = scan_odd_valid;
                ws1_wr_addr  = scan_idx;
                ws1_wr_data  = '0;
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
            score_read_valid   <= 1'b0;
            score_read_data    <= '0;
            score_read_pending <= 1'b0;
            score_read_bank    <= 1'b0;
        end else begin
            state         <= state_next;
            cycle_count   <= cycle_count + 32'd1;
            neuron_spikes <= '0;
            score_read_valid <= score_read_pending;
            if (score_read_pending)
                score_read_data <= score_read_bank ? mem1_rd_data : mem0_rd_data;
            score_read_pending <= (state == S_IDLE) && score_read_en;
            score_read_bank    <= score_read_id[0];

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
                        clear_idx <= clear_idx + BANK_ADDR_WIDTH'(1);
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
                    if (even_fired && !is_output_id(scan_even_id))
                        neuron_spikes[scan_even_id] <= 1'b1;
                    if (scan_odd_push && !is_output_id(scan_odd_id))
                        neuron_spikes[scan_odd_id] <= 1'b1;

                    total_spikes_fired <= total_spikes_fired + {30'd0, scan_fire_count};
                    if (!scan_both_fire && !scan_last)
                        scan_idx <= scan_idx + BANK_ADDR_WIDTH'(1);
                end

                S_SCAN_PUSH_ODD: begin
                    if (!scan_last)
                        scan_idx <= scan_idx + BANK_ADDR_WIDTH'(1);
                end

                default: ;
            endcase
        end
    end

endmodule
