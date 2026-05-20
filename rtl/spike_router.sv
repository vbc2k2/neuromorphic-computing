`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// spike_router.sv — CSR-Based Sparse Spike Router
//
// Description:
//   Buffers spike events (source neuron ids) in a FIFO and fans each one out
//   to its destination neurons. Connectivity comes from a CSR store, so the
//   router walks ONLY the real synapses of a spiking neuron — O(fan-out) work,
//   never O(N). This is the core event-driven efficiency mechanism: routing
//   cost is proportional to spike activity, not to network size.
//
// Parameters:
//   NUM_NEURONS  — number of neurons
//   WEIGHT_WIDTH — weight bit width
//   NUM_SYN      — total number of synapses in the CSR store
//   FIFO_DEPTH   — spike event FIFO depth
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

module spike_router #(
    parameter int NUM_NEURONS  = 139,
    parameter int WEIGHT_WIDTH = 8,
    parameter int NUM_SYN      = 4810,
    parameter int FIFO_DEPTH   = 256
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Spike input — source neuron ids to fan out
    input  logic [$clog2(NUM_NEURONS)-1:0] spike_in_id,
    input  logic                        spike_in_valid,
    output logic                        spike_in_ready,

    // CSR connectivity interface
    output logic [$clog2(NUM_NEURONS)-1:0] csr_ptr_src,
    input  logic [$clog2(NUM_SYN+1)-1:0]   csr_ptr_base,
    input  logic [$clog2(NUM_SYN+1)-1:0]   csr_ptr_len,
    output logic [$clog2(NUM_SYN+1)-1:0]   csr_syn_index,
    output logic                           csr_syn_rd_en,
    input  logic [$clog2(NUM_NEURONS)-1:0] csr_syn_dst,
    input  logic signed [WEIGHT_WIDTH-1:0] csr_syn_weight,
    input  logic                           csr_syn_valid,

    // Delivery to neuron array
    output logic [$clog2(NUM_NEURONS)-1:0] deliver_dst_id,
    output logic signed [WEIGHT_WIDTH-1:0] deliver_weight,
    output logic                        deliver_valid,

    // Status
    output logic                        busy,
    output logic                        idle,             // IDLE and FIFO empty
    output logic [31:0]                 total_events,     // source spikes processed
    output logic [31:0]                 total_deliveries  // synapse deliveries
);

    localparam int ID_WIDTH  = $clog2(NUM_NEURONS);
    localparam int IDX_WIDTH = $clog2(NUM_SYN + 1);

    // =========================================================================
    // FIFO of spike events (source neuron ids)
    // =========================================================================
    localparam int FIFO_COUNT_WIDTH = $clog2(FIFO_DEPTH) + 1;

    logic [ID_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];
    logic [$clog2(FIFO_DEPTH)-1:0] fifo_wr_ptr, fifo_rd_ptr;
    logic [FIFO_COUNT_WIDTH-1:0]   fifo_count;

    logic fifo_full, fifo_empty;
    assign fifo_full      = (fifo_count == FIFO_COUNT_WIDTH'(FIFO_DEPTH));
    assign fifo_empty     = (fifo_count == 0);
    assign spike_in_ready = !fifo_full;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fifo_wr_ptr <= '0;
        else if (spike_in_valid && spike_in_ready) begin
            fifo_mem[fifo_wr_ptr] <= spike_in_id;
            fifo_wr_ptr <= fifo_wr_ptr + 1;
        end
    end

    // =========================================================================
    // Fan-out FSM — walk the current source's CSR synapse list
    // =========================================================================
    typedef enum logic [1:0] {
        IDLE,
        DEQUEUE,        // pop a source id; CSR pointer becomes valid next cycle
        CHECK,          // base/len ready; skip sources with no fan-out
        STREAM          // pipelined read/deliver of all fan-out synapses
    } state_t;

    state_t state, state_next;
    logic [ID_WIDTH-1:0]  current_src;
    logic [IDX_WIDTH-1:0] issue_idx;
    logic [IDX_WIDTH-1:0] deliver_idx;
    logic                 issue_valid;
    logic                 delivery_valid;
    logic                 last_delivery;

    assign issue_valid    = (state == STREAM) && (issue_idx < csr_ptr_len);
    assign delivery_valid = (state == STREAM) && csr_syn_valid;
    assign last_delivery  = delivery_valid && (deliver_idx + IDX_WIDTH'(1) == csr_ptr_len);
    assign busy     = (state != IDLE);
    assign idle     = (state == IDLE) && fifo_empty;

    // CSR pointer is looked up for the source currently being processed
    assign csr_ptr_src   = current_src;
    assign csr_syn_index = csr_ptr_base + issue_idx;
    assign csr_syn_rd_en = issue_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= state_next;
    end

    always_comb begin
        state_next = state;
        case (state)
            IDLE:     if (!fifo_empty) state_next = DEQUEUE;
            DEQUEUE:  state_next = CHECK;
            CHECK:    state_next = (csr_ptr_len == 0) ? IDLE : STREAM;
            STREAM:   if (last_delivery) state_next = IDLE;
            default:  state_next = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_src      <= '0;
            issue_idx        <= '0;
            deliver_idx      <= '0;
            fifo_rd_ptr      <= '0;
            fifo_count       <= '0;
            total_events     <= '0;
            total_deliveries <= '0;
        end else begin
            // FIFO occupancy
            case ({spike_in_valid && spike_in_ready, (state == DEQUEUE)})
                2'b10:   fifo_count <= fifo_count + 1;
                2'b01:   fifo_count <= fifo_count - 1;
                default: ;
            endcase

            case (state)
                DEQUEUE: begin
                    current_src  <= fifo_mem[fifo_rd_ptr];
                    fifo_rd_ptr  <= fifo_rd_ptr + 1;
                    issue_idx    <= '0;
                    deliver_idx  <= '0;
                    total_events <= total_events + 1;
                end
                STREAM: begin
                    if (issue_valid)
                        issue_idx <= issue_idx + 1;
                    if (delivery_valid) begin
                        deliver_idx      <= deliver_idx + 1;
                        total_deliveries <= total_deliveries + 1;
                    end
                end
                default: ;
            endcase
        end
    end

    // Delivery outputs - one synapse per cycle after the first CSR latency
    assign deliver_dst_id = csr_syn_dst;
    assign deliver_weight = csr_syn_weight;
    assign deliver_valid  = delivery_valid;

endmodule
