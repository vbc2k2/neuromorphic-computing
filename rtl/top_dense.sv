`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top_dense.sv — Dense Clock-Driven SNN Baseline
//
// Description:
//   The honest dense baseline for the SNN classifier comparison. Every
//   timestep it walks EVERY synapse in the network and performs the
//   multiply-accumulate, whether or not the presynaptic neuron actually
//   spiked. This is the spiking equivalent of a plain ANN accelerator that
//   recomputes every layer every frame.
//
//   Work per timestep is constant and data-independent:
//       synapse_ops = NUM_SYN   (every synapse, every timestep)
//   The event-driven accelerator (top.sv) computes the SAME result but only
//   walks the synapses of neurons that actually spiked.
//
//   Uses the SAME neuron_core and SAME CSR connectivity as top.sv, so the two
//   designs are functionally equivalent by construction.
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

module top_dense #(
    parameter int NUM_NEURONS    = 139,
    parameter int WEIGHT_WIDTH   = 8,
    parameter int MEMBRANE_WIDTH = 16,
    parameter int THRESHOLD      = 210,
    parameter int LEAK           = 1,
    parameter int NUM_SYN        = 4810,
    parameter     CSR_DST_FILE    = "csr_dst.mem",
    parameter     CSR_WEIGHT_FILE = "csr_weight.mem",
    parameter     CSR_START_FILE  = "csr_start.mem",
    parameter     CSR_COUNT_FILE  = "csr_count.mem"
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // External spike injection (OR-ed into the active spike vector)
    input  logic [NUM_NEURONS-1:0]      ext_spike_vector,
    input  logic                        ext_spike_load,

    // Timestep handshake
    input  logic                        start_step,
    output logic                        step_busy,

    // Output spike monitoring
    output logic [NUM_NEURONS-1:0]      neuron_spikes,
    output logic signed [MEMBRANE_WIDTH-1:0] debug_membrane [NUM_NEURONS],

    // Performance counters
    output logic [31:0]                 cycle_count,
    output logic [31:0]                 active_cycles,
    output logic [31:0]                 total_spikes_fired,
    output logic [31:0]                 total_updates,
    output logic [31:0]                 total_synapse_ops
);

    localparam int ID_WIDTH  = $clog2(NUM_NEURONS);
    localparam int IDX_WIDTH = $clog2(NUM_SYN + 1);

    // =========================================================================
    // CSR Synapse Connectivity Store (same store as the event-driven design)
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
    // Active spike vector
    // =========================================================================
    logic [NUM_NEURONS-1:0] spike_vector;
    logic [NUM_NEURONS-1:0] fired_r;
    logic [NUM_NEURONS-1:0] fired_comb;

    // =========================================================================
    // Dense scanner FSM — walks every source's full synapse list each timestep
    // =========================================================================
    typedef enum logic [3:0] {
        D_IDLE, D_SRC, D_CHECK, D_READ, D_WAIT, D_PROC, D_NEXT_SYN,
        D_NEXT_SRC, D_FIRE, D_LATCH, D_CAPTURE
    } d_state_t;

    d_state_t             state, state_next;
    logic [ID_WIDTH-1:0]  scan_src;
    logic [IDX_WIDTH-1:0] walk_idx;
    logic                 timestep_tick;
    logic                 last_syn, last_src;

    assign neuron_spikes = fired_comb;
    assign step_busy     = (state != D_IDLE);
    assign timestep_tick = (state == D_FIRE);

    assign csr_ptr_src   = scan_src;
    assign csr_syn_index = csr_ptr_base + walk_idx;
    assign csr_syn_rd_en = (state == D_READ);

    assign last_syn = (walk_idx + 1 == csr_ptr_len);
    assign last_src = (scan_src == ID_WIDTH'(NUM_NEURONS - 1));

    // Popcount of neurons firing this timestep
    integer fire_sum;
    always_comb begin
        fire_sum = 0;
        for (int i = 0; i < NUM_NEURONS; i++)
            if (fired_comb[i])
                fire_sum = fire_sum + 1;
    end

    // Next-state logic
    always_comb begin
        state_next = state;
        case (state)
            D_IDLE:     if (start_step) state_next = D_SRC;
            D_SRC:                      state_next = D_CHECK;
            D_CHECK:    state_next = (csr_ptr_len == 0) ? D_NEXT_SRC : D_READ;
            D_READ:                     state_next = D_WAIT;
            D_WAIT:     if (csr_syn_valid) state_next = D_PROC;
            D_PROC:                     state_next = D_NEXT_SYN;
            D_NEXT_SYN: state_next = last_syn ? D_NEXT_SRC : D_READ;
            D_NEXT_SRC: state_next = last_src ? D_FIRE : D_SRC;
            D_FIRE:                     state_next = D_LATCH;
            D_LATCH:                    state_next = D_CAPTURE;
            D_CAPTURE:                  state_next = D_IDLE;
            default:                    state_next = D_IDLE;
        endcase
    end

    // State register and datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= D_IDLE;
            scan_src           <= '0;
            walk_idx           <= '0;
            spike_vector       <= '0;
            fired_r            <= '0;
            cycle_count        <= '0;
            active_cycles      <= '0;
            total_spikes_fired <= '0;
            total_updates      <= '0;
            total_synapse_ops  <= '0;
        end else begin
            state       <= state_next;
            cycle_count <= cycle_count + 1;
            if (state != D_IDLE)
                active_cycles <= active_cycles + 1;

            case (state)
                D_IDLE: begin
                    if (ext_spike_load)
                        spike_vector <= spike_vector | ext_spike_vector;
                    if (start_step)
                        scan_src <= '0;
                end
                D_SRC: begin
                    walk_idx <= '0;
                end
                D_PROC: begin
                    // dense baseline: every synapse is a counted operation
                    total_synapse_ops <= total_synapse_ops + 1;
                end
                D_NEXT_SYN: begin
                    if (!last_syn)
                        walk_idx <= walk_idx + 1;
                end
                D_NEXT_SRC: begin
                    if (!last_src)
                        scan_src <= scan_src + 1;
                end
                D_FIRE: begin
                    total_updates <= total_updates + NUM_NEURONS;
                end
                D_LATCH: begin
                    fired_r            <= fired_comb;
                    total_spikes_fired <= total_spikes_fired + 32'(fire_sum);
                end
                D_CAPTURE: begin
                    spike_vector <= fired_r;
                end
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Neuron Array — shared neuron_core, identical to the event-driven design
    // =========================================================================
    genvar g;
    generate
        for (g = 0; g < NUM_NEURONS; g++) begin : gen_neurons
            logic neuron_spike_in_valid;
            // deliver weight to its target neuron when the source spiked
            assign neuron_spike_in_valid =
                (state == D_PROC) && (csr_syn_dst == ID_WIDTH'(g))
                && spike_vector[scan_src];

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
                .spike_weight       (csr_syn_weight),
                .timestep_tick      (timestep_tick),
                .spike_out          (fired_comb[g]),
                .membrane_potential (debug_membrane[g])
            );
        end
    endgenerate

endmodule
