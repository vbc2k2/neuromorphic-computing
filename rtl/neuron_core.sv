`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// neuron_core.sv — Unified Discrete-Timestep Leaky Integrate-and-Fire Neuron
//
// Description:
//   A single LIF neuron, shared by BOTH the event-driven and the time-stepped
//   accelerators. Sharing one neuron module is what makes the two designs
//   functionally equivalent by construction — they differ only in *how* the
//   synaptic weights are delivered, never in the neuron math.
//
//   Per timestep the neuron computes the standard LIF update:
//       m_next = leak(m) + weight_sum          (saturation-free in normal range)
//       if (m_next >= THRESHOLD)  -> fire, m_next = RESET_VAL
//
//   Interface:
//     - spike_in_valid / spike_weight : a weighted synaptic input is delivered.
//       Multiple deliveries within a timestep simply accumulate (commutative),
//       so delivery order does not matter.
//     - timestep_tick : end-of-timestep pulse. Applies leak, integrates the
//       accumulated weight_sum, performs the threshold check, then clears the
//       accumulator for the next timestep.
//
// Parameters:
//   WEIGHT_WIDTH    — bit width of synaptic weights (signed)
//   MEMBRANE_WIDTH  — bit width of membrane potential (signed)
//   THRESHOLD       — firing threshold
//   LEAK            — leak amount subtracted each timestep
//   RESET_VAL       — membrane value after firing
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

module neuron_core #(
    parameter int WEIGHT_WIDTH   = 8,
    parameter int MEMBRANE_WIDTH = 16,
    parameter int THRESHOLD      = 25,
    parameter int LEAK           = 1,
    parameter int RESET_VAL      = 0
)(
    input  logic                             clk,
    input  logic                             rst_n,

    // Synaptic input delivery — accumulates into weight_sum within a timestep
    input  logic                             spike_in_valid,
    input  logic signed [WEIGHT_WIDTH-1:0]    spike_weight,

    // End-of-timestep pulse — leak + integrate + threshold check
    input  logic                             timestep_tick,

    // Outputs
    output logic                             spike_out,            // 1-cycle pulse
    output logic signed [MEMBRANE_WIDTH-1:0]  membrane_potential
);

    // State: membrane potential and the per-timestep weight accumulator
    logic signed [MEMBRANE_WIDTH-1:0] membrane_r;
    logic signed [MEMBRANE_WIDTH-1:0] weight_sum_r;
    logic                             fire_r;

    assign spike_out          = fire_r;
    assign membrane_potential = membrane_r;

    // Combinational timestep update (leak -> integrate -> threshold)
    logic signed [MEMBRANE_WIDTH-1:0] leaked;
    logic signed [MEMBRANE_WIDTH-1:0] integrated;
    logic                             fired_c;

    always_comb begin
        // Leak: decay toward resting potential, never crossing zero from above
        if (membrane_r > MEMBRANE_WIDTH'(signed'(LEAK)))
            leaked = membrane_r - MEMBRANE_WIDTH'(signed'(LEAK));
        else if (membrane_r > 0)
            leaked = '0;
        else
            leaked = membrane_r;  // already at/below resting potential

        // Integrate this timestep's accumulated synaptic input
        integrated = leaked + weight_sum_r;

        // Threshold check
        fired_c = (integrated >= MEMBRANE_WIDTH'(signed'(THRESHOLD)));
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_r   <= '0;
            weight_sum_r <= '0;
            fire_r       <= 1'b0;
        end else if (timestep_tick) begin
            // End of timestep: commit the LIF update
            if (fired_c) begin
                membrane_r <= MEMBRANE_WIDTH'(signed'(RESET_VAL));
                fire_r     <= 1'b1;
            end else begin
                membrane_r <= integrated;
                fire_r     <= 1'b0;
            end
            weight_sum_r <= '0;             // clear accumulator for next timestep
        end else begin
            fire_r <= 1'b0;                 // spike_out is a 1-cycle pulse
            if (spike_in_valid)
                weight_sum_r <= weight_sum_r + MEMBRANE_WIDTH'(spike_weight);
        end
    end

endmodule
