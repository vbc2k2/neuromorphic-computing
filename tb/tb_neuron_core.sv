//-----------------------------------------------------------------------------
// tb_neuron_core.sv — Neuron Core Unit Testbench
//
// Tests the unified discrete-timestep LIF neuron:
//   1. Synaptic accumulation into the per-timestep weight sum
//   2. Threshold firing and membrane reset on timestep_tick
//   3. Leak decay on timesteps with no input
//   4. Sub-threshold behaviour (no fire)
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_neuron_core;

    // Parameters matching DUT
    localparam int WEIGHT_WIDTH   = 8;
    localparam int MEMBRANE_WIDTH = 16;
    localparam int THRESHOLD      = 100;
    localparam int LEAK           = 2;
    localparam int RESET_VAL      = 0;

    // Clock and reset
    logic clk;
    logic rst_n;

    // DUT signals
    logic                             spike_in_valid;
    logic signed [WEIGHT_WIDTH-1:0]    spike_weight;
    logic                             timestep_tick;
    logic                             spike_out;
    logic signed [MEMBRANE_WIDTH-1:0]  membrane_potential;

    // Instantiate DUT
    neuron_core #(
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .MEMBRANE_WIDTH (MEMBRANE_WIDTH),
        .THRESHOLD      (THRESHOLD),
        .LEAK           (LEAK),
        .RESET_VAL      (RESET_VAL)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .spike_in_valid     (spike_in_valid),
        .spike_weight       (spike_weight),
        .timestep_tick      (timestep_tick),
        .spike_out          (spike_out),
        .membrane_potential (membrane_potential)
    );

    // Clock generation: 10ns period (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // Test tracking
    int test_num   = 0;
    int pass_count = 0;
    int fail_count = 0;

    // Task: check
    task automatic check(string msg, logic condition);
        test_num++;
        if (condition) begin
            $display("[PASS] Test %0d: %s", test_num, msg);
            pass_count++;
        end else begin
            $display("[FAIL] Test %0d: %s", test_num, msg);
            fail_count++;
        end
    endtask

    // Task: deliver one weighted synaptic input (accumulates into weight sum)
    task automatic deliver(input logic signed [WEIGHT_WIDTH-1:0] weight);
        @(posedge clk);
        spike_in_valid <= 1'b1;
        spike_weight   <= weight;
        @(posedge clk);
        spike_in_valid <= 1'b0;
        spike_weight   <= '0;
    endtask

    // Task: end-of-timestep tick (leak + integrate + threshold check)
    task automatic tick();
        @(posedge clk);
        timestep_tick <= 1'b1;
        @(posedge clk);
        timestep_tick <= 1'b0;
        #1;  // let the registered update settle before checking
    endtask

    // VCD dump
    initial begin
        $dumpfile("tb_neuron_core.vcd");
        $dumpvars(0, tb_neuron_core);
    end

    // Main test sequence
    initial begin
        $display("==========================================================");
        $display("  Neuron Core Unit Testbench");
        $display("  WEIGHT_WIDTH=%0d, MEMBRANE_WIDTH=%0d, THRESHOLD=%0d, LEAK=%0d",
                 WEIGHT_WIDTH, MEMBRANE_WIDTH, THRESHOLD, LEAK);
        $display("==========================================================");

        // Initialize
        rst_n          = 1'b0;
        spike_in_valid = 1'b0;
        spike_weight   = '0;
        timestep_tick  = 1'b0;

        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        // =====================================================================
        // TEST 1: Membrane accumulation
        // =====================================================================
        $display("\n--- Test Group 1: Membrane Accumulation ---");

        deliver(8'sd20);
        deliver(8'sd30);
        deliver(8'sd40);     // weight sum accumulated = 90
        tick();              // membrane = leak(0) + 90 = 90
        check("Membrane = 90 after accumulating 20+30+40 and a tick",
              membrane_potential == 16'sd90);
        check("No fire below threshold", spike_out == 1'b0);

        // =====================================================================
        // TEST 2: Threshold firing and reset
        // =====================================================================
        $display("\n--- Test Group 2: Threshold Fire & Reset ---");

        deliver(8'sd40);     // leak(90)=88, +40 = 128 >= 100 -> fire
        tick();
        check("Membrane reset and spike pulse asserted on threshold crossing",
              membrane_potential == 16'sd0 && spike_out == 1'b1);
        @(posedge clk); #1;
        check("Spike pulse deasserts after one cycle", spike_out == 1'b0);

        // =====================================================================
        // TEST 3: Leak decay
        // =====================================================================
        $display("\n--- Test Group 3: Leak Decay ---");

        deliver(8'sd20);
        tick();              // membrane = leak(0) + 20 = 20
        check("Membrane = 20 after +20 spike and tick", membrane_potential == 16'sd20);

        tick();              // no input: membrane = leak(20) = 18
        check("Membrane = 18 after leak tick (20-2)", membrane_potential == 16'sd18);

        tick();              // no input: membrane = leak(18) = 16
        check("Membrane = 16 after second leak tick (18-2)", membrane_potential == 16'sd16);

        // =====================================================================
        // TEST 4: Sub-threshold (no fire)
        // =====================================================================
        $display("\n--- Test Group 4: Sub-threshold (No Fire) ---");

        rst_n = 1'b0;
        repeat(3) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        deliver(8'sd99);     // just below threshold of 100
        tick();
        check("Membrane = 99 (below threshold, no fire)",
              membrane_potential == 16'sd99 && spike_out == 1'b0);

        // =====================================================================
        // Summary
        // =====================================================================
        $display("\n==========================================================");
        $display("  Results: %0d PASSED, %0d FAILED out of %0d tests",
                 pass_count, fail_count, test_num);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** SOME TESTS FAILED ***");
        $display("==========================================================\n");

        #100;
        $finish;
    end

endmodule
