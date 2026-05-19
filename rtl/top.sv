`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top.sv — Event-Driven SNN Accelerator Top-Level
//
// Description:
//   Integrates N shared neuron cores, a CSR synapse-connectivity store, and a
//   sparse spike router into an event-driven SNN accelerator.
//
//   Discrete-timestep operation (handshaked by the testbench):
//     1. start_step begins a timestep.
//     2. ROUTE   : the router drains its spike FIFO (external injections plus
//                  neurons that fired last timestep). For each spiking source
//                  it walks ONLY that neuron's CSR fan-out list, delivering a
//                  weighted input to each target neuron.
//     3. FIRE    : a global timestep_tick makes every neuron leak, integrate
//                  its accumulated input, and threshold-check at once.
//     4. CAPTURE : neurons that fired are queued as the source events for the
//                  next timestep.
//
//   Work is proportional to spike activity, not to network size — the same
//   result as the dense baseline (top_dense.sv), reached with far less work.
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

module top #(
    parameter int NUM_NEURONS    = 139,
    parameter int WEIGHT_WIDTH   = 8,
    parameter int MEMBRANE_WIDTH = 16,
    parameter int THRESHOLD      = 210,
    parameter int LEAK           = 1,
    parameter int NUM_SYN        = 4810,
    parameter int FIFO_DEPTH     = 256,
    parameter     CSR_DST_FILE    = "csr_dst.mem",
    parameter     CSR_WEIGHT_FILE = "csr_weight.mem",
    parameter     CSR_START_FILE  = "csr_start.mem",
    parameter     CSR_COUNT_FILE  = "csr_count.mem"
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // External spike injection (accepted while idle, before start_step)
    input  logic [$clog2(NUM_NEURONS)-1:0] ext_spike_id,
    input  logic                        ext_spike_valid,

    // Timestep handshake
    input  logic                        start_step,
    output logic                        step_busy,

    // Output spike monitoring
    output logic [NUM_NEURONS-1:0]      neuron_spikes,
    output logic signed [MEMBRANE_WIDTH-1:0] debug_membrane [NUM_NEURONS],

    // Performance counters
    output logic [31:0]                 cycle_count,        // free-running
    output logic [31:0]                 active_cycles,      // cycles doing work
    output logic [31:0]                 total_spikes_fired,
    output logic [31:0]                 router_events,      // source spikes routed
    output logic [31:0]                 router_deliveries   // synapse operations
);

    localparam int ID_WIDTH  = $clog2(NUM_NEURONS);
    localparam int IDX_WIDTH = $clog2(NUM_SYN + 1);

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
    // Spike Router
    // =========================================================================
    logic [ID_WIDTH-1:0]            router_spike_id;
    logic                           router_spike_valid;
    logic                           router_spike_ready;

    logic [ID_WIDTH-1:0]            deliver_dst_id;
    logic signed [WEIGHT_WIDTH-1:0] deliver_weight;
    logic                           deliver_valid;
    logic                           router_busy;
    logic                           router_idle;

    spike_router #(
        .NUM_NEURONS  (NUM_NEURONS),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .NUM_SYN      (NUM_SYN),
        .FIFO_DEPTH   (FIFO_DEPTH)
    ) u_spike_router (
        .clk              (clk),
        .rst_n            (rst_n),
        .spike_in_id      (router_spike_id),
        .spike_in_valid   (router_spike_valid),
        .spike_in_ready   (router_spike_ready),
        .csr_ptr_src      (csr_ptr_src),
        .csr_ptr_base     (csr_ptr_base),
        .csr_ptr_len      (csr_ptr_len),
        .csr_syn_index    (csr_syn_index),
        .csr_syn_rd_en    (csr_syn_rd_en),
        .csr_syn_dst      (csr_syn_dst),
        .csr_syn_weight   (csr_syn_weight),
        .csr_syn_valid    (csr_syn_valid),
        .deliver_dst_id   (deliver_dst_id),
        .deliver_weight   (deliver_weight),
        .deliver_valid    (deliver_valid),
        .busy             (router_busy),
        .idle             (router_idle),
        .total_events     (router_events),
        .total_deliveries (router_deliveries)
    );

    // =========================================================================
    // Timestep FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,     // accept external spikes, wait for start_step
        S_ROUTE,    // router drains the spike FIFO
        S_FIRE,     // global timestep_tick: leak + integrate + threshold
        S_LATCH,    // neuron spike_out pulses become valid
        S_CAPTURE   // feed fired neurons back into the router FIFO
    } state_t;

    state_t state, state_next;

    logic [NUM_NEURONS-1:0] neuron_fired;
    logic [NUM_NEURONS-1:0] cap_bits;        // fired neurons pending feedback
    logic                   timestep_tick;

    assign neuron_spikes = neuron_fired;
    assign step_busy     = (state != S_IDLE);
    assign timestep_tick = (state == S_FIRE);

    // Lowest pending fired neuron (priority encoder over cap_bits)
    logic [ID_WIDTH-1:0] cap_id;
    logic                cap_has;
    always_comb begin
        cap_has = 1'b0;
        cap_id  = '0;
        for (int i = 0; i < NUM_NEURONS; i++) begin
            if (cap_bits[i] && !cap_has) begin
                cap_has = 1'b1;
                cap_id  = ID_WIDTH'(i);
            end
        end
    end

    // Popcount of neurons firing this timestep
    integer fire_sum;
    always_comb begin
        fire_sum = 0;
        for (int i = 0; i < NUM_NEURONS; i++)
            if (neuron_fired[i])
                fire_sum = fire_sum + 1;
    end

    // Next-state logic
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:    if (start_step)   state_next = S_ROUTE;
            S_ROUTE:   if (router_idle)  state_next = S_FIRE;
            S_FIRE:                      state_next = S_LATCH;
            S_LATCH:                     state_next = S_CAPTURE;
            S_CAPTURE: if (!cap_has)      state_next = S_IDLE;
            default:                     state_next = S_IDLE;
        endcase
    end

    // Router enqueue mux: external spikes while idle, feedback while capturing
    always_comb begin
        router_spike_id    = ext_spike_id;
        router_spike_valid = 1'b0;
        if (state == S_IDLE) begin
            router_spike_id    = ext_spike_id;
            router_spike_valid = ext_spike_valid;
        end else if (state == S_CAPTURE) begin
            router_spike_id    = cap_id;
            router_spike_valid = cap_has;
        end
    end

    // State register and datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            cap_bits           <= '0;
            cycle_count        <= '0;
            active_cycles      <= '0;
            total_spikes_fired <= '0;
        end else begin
            state       <= state_next;
            cycle_count <= cycle_count + 1;

            if (state != S_IDLE)
                active_cycles <= active_cycles + 1;

            case (state)
                S_LATCH: begin
                    cap_bits           <= neuron_fired;
                    total_spikes_fired <= total_spikes_fired + 32'(fire_sum);
                end
                S_CAPTURE: begin
                    if (cap_has && router_spike_ready)
                        cap_bits[cap_id] <= 1'b0;
                end
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Neuron Array — shared neuron_core, identical to the dense baseline
    // =========================================================================
    genvar g;
    generate
        for (g = 0; g < NUM_NEURONS; g++) begin : gen_neurons
            logic neuron_spike_in_valid;
            assign neuron_spike_in_valid =
                deliver_valid && (deliver_dst_id == ID_WIDTH'(g));

            neuron_core #(
                .WEIGHT_WIDTH   (WEIGHT_WIDTH),
                .MEMBRANE_WIDTH (MEMBRANE_WIDTH),
                .THRESHOLD      (THRESHOLD),
                .LEAK           (LEAK),
                .RESET_VAL      (0)
            ) u_neuron (
                .clk                (clk),
                .rst_n              (rst_n),
                .spike_in_valid     (neuron_spike_in_valid),
                .spike_weight       (deliver_weight),
                .timestep_tick      (timestep_tick),
                .spike_out          (neuron_fired[g]),
                .membrane_potential (debug_membrane[g])
            );
        end
    endgenerate

endmodule
