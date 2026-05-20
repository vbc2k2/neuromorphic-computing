//-----------------------------------------------------------------------------
// tb_classify_dense.sv — Dense-Baseline SNN Digit Classification Testbench
//
// Description:
//   Runs the dense clock-driven baseline (top_dense) as a digit classifier on
//   the SAME images and SAME network as tb_classify.sv. Because both designs
//   share the neuron core and weights, they must produce identical predictions
//   and identical accuracy -- this testbench verifies that, and measures the
//   dense baseline's (much larger) synaptic-operation count.
//
// Author: Neuromorphic Accelerator Project
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "snn_config.vh"

module tb_classify_dense;

    localparam int N        = `SNN_N_TOTAL;
    localparam int N_INPUT  = `SNN_N_INPUT;
    localparam int N_OUTPUT = `SNN_N_OUTPUT;
    localparam int OUT_BASE  = `SNN_ID_OUTPUT_BASE;
    localparam int ID_BIAS   = `SNN_ID_BIAS;
    localparam int T_STEPS   = `SNN_T_STEPS;
    localparam int THRESHOLD = `SNN_THRESHOLD;
    localparam int LEAK      = `SNN_LEAK;
    localparam int NUM_TEST  = `SNN_NUM_TEST;

    localparam int NUM_SYN        = `SNN_NUM_SYN;
    localparam int WEIGHT_WIDTH   = 8;
    localparam int MEMBRANE_WIDTH = 16;

    // Clock and reset
    logic clk;
    logic rst_n;

    // DUT interface
    logic [N-1:0]           ext_spike_vector;
    logic                   ext_spike_load;
    logic                   start_step;
    logic                   step_busy;
    logic [N-1:0]           neuron_spikes;
    logic signed [MEMBRANE_WIDTH-1:0] debug_membrane [N];
    logic [31:0]            cycle_count;
    logic [31:0]            active_cycles;
    logic [31:0]            total_spikes_fired;
    logic [31:0]            total_updates;
    logic [31:0]            total_synapse_ops;

    top_dense #(
        .NUM_NEURONS    (N),
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .MEMBRANE_WIDTH (MEMBRANE_WIDTH),
        .THRESHOLD      (THRESHOLD),
        .LEAK           (LEAK),
        .NUM_SYN        (NUM_SYN)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .ext_spike_vector   (ext_spike_vector),
        .ext_spike_load     (ext_spike_load),
        .start_step         (start_step),
        .step_busy          (step_busy),
        .neuron_spikes      (neuron_spikes),
        .debug_membrane     (debug_membrane),
        .cycle_count        (cycle_count),
        .active_cycles      (active_cycles),
        .total_spikes_fired (total_spikes_fired),
        .total_updates      (total_updates),
        .total_synapse_ops  (total_synapse_ops)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Stimulus memories
    logic [N_INPUT-1:0] spike_mem [0:NUM_TEST*T_STEPS-1];
    logic [3:0]         label_mem [0:NUM_TEST-1];

    // Output-spike counters
    int  out_count [0:N_OUTPUT-1];
    bit  count_en;

    always @(posedge clk) begin
        if (count_en) begin
            for (int o = 0; o < N_OUTPUT; o++)
                if (neuron_spikes[OUT_BASE + o])
                    out_count[o] <= out_count[o] + 1;
        end
    end

    longint tot_active, tot_synops, tot_updates, tot_spikes;
    int     correct;
    integer result_log;

    // Task: build and load this timestep's input spike vector
    task automatic load_inputs(input int img, input int t);
        logic [N-1:0]       vec;
        logic [N_INPUT-1:0] word;
        vec  = '0;
        word = spike_mem[img*T_STEPS + t];
        for (int p = 0; p < N_INPUT; p++)
            if (word[p]) vec[p] = 1'b1;
        vec[ID_BIAS] = 1'b1;                  // bias neuron fires every timestep
        @(posedge clk);
        ext_spike_vector <= vec;
        ext_spike_load   <= 1'b1;
        @(posedge clk);
        ext_spike_load   <= 1'b0;
        ext_spike_vector <= '0;
    endtask

    // Task: run one timestep
    task automatic run_step();
        @(posedge clk);
        start_step <= 1'b1;
        @(posedge clk);
        start_step <= 1'b0;
        while (!step_busy) @(posedge clk);
        while (step_busy)  @(posedge clk);
        @(posedge clk);
    endtask

    // Task: reset the network for a new image
    task automatic reset_dut();
        rst_n <= 1'b0;
        repeat(4) @(posedge clk);
        rst_n <= 1'b1;
        repeat(2) @(posedge clk);
    endtask

    // Task: classify one image
    task automatic classify(input int img, output int prediction);
        int best;
        reset_dut();
        for (int o = 0; o < N_OUTPUT; o++) out_count[o] = 0;
        count_en = 1'b1;
        for (int t = 0; t < T_STEPS; t++) begin
            load_inputs(img, t);
            run_step();
        end
        count_en = 1'b0;
        @(posedge clk);
        prediction = 0;
        best = out_count[0];
        for (int o = 1; o < N_OUTPUT; o++)
            if (out_count[o] > best) begin
                best = out_count[o];
                prediction = o;
            end
    endtask

    // Image range + output tag — overridable via plusargs for sbatch slicing:
    //   xsim/xrun ... +first=<N> +last=<M> +tag=<suffix>
    int    run_first, run_last, nrun;
    string tag;

    initial begin
        int pred;
        if (!$value$plusargs("first=%d", run_first)) run_first = 0;
        if (!$value$plusargs("last=%d",  run_last))  run_last  = NUM_TEST - 1;
        if (!$value$plusargs("tag=%s",   tag))       tag       = "";
        if (run_last > NUM_TEST - 1) run_last = NUM_TEST - 1;
        nrun = run_last - run_first + 1;

        $display("==========================================================");
        $display("  Dense Baseline SNN Accelerator - Digit Classification");
        $display("  %0d neurons, threshold=%0d, %0d timesteps/image",
                 N, THRESHOLD, T_STEPS);
        $display("  images %0d..%0d", run_first, run_last);
        $display("==========================================================");

        $readmemb("snn_spikes.mem", spike_mem);
        $readmemh("snn_labels.mem", label_mem);

        result_log = $fopen($sformatf("classify_dense%s.csv", tag), "w");
        $fwrite(result_log, "image,label,prediction,correct,active_cycles,synapse_ops,spikes\n");

        rst_n            = 1'b1;
        ext_spike_vector = '0;
        ext_spike_load   = 1'b0;
        start_step       = 1'b0;
        count_en         = 1'b0;
        correct          = 0;
        tot_active = 0; tot_synops = 0; tot_updates = 0; tot_spikes = 0;
        repeat(5) @(posedge clk);

        for (int img = run_first; img <= run_last; img++) begin
            classify(img, pred);
            if (pred == int'(label_mem[img])) correct++;
            tot_active  += longint'(active_cycles);
            tot_synops  += longint'(total_synapse_ops);
            tot_updates += longint'(total_updates);
            tot_spikes  += longint'(total_spikes_fired);
            $fwrite(result_log, "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    img, label_mem[img], pred,
                    (pred == int'(label_mem[img])) ? 1 : 0,
                    active_cycles, total_synapse_ops, total_spikes_fired);
            if (img - run_first < 12)
                $display("  image %0d: label=%0d predict=%0d  %s",
                         img, label_mem[img], pred,
                         (pred == int'(label_mem[img])) ? "OK" : "x");
        end

        $display("\n==========================================================");
        $display("  Classification accuracy: %0d / %0d = %0d.%02d%%",
                 correct, nrun, (correct*100)/nrun,
                 ((correct*10000)/nrun) % 100);
        $display("  Dense-baseline work:");
        $display("    Active cycles:       %0d", tot_active);
        $display("    Synapse ops:         %0d", tot_synops);
        $display("    Neuron updates:      %0d", tot_updates);
        $display("    Spikes fired:        %0d", tot_spikes);
        $display("==========================================================\n");

        $fclose(result_log);

        result_log = $fopen($sformatf("metrics_classify_dense%s.csv", tag), "w");
        $fwrite(result_log, "metric,value\n");
        $fwrite(result_log, "design,dense_baseline\n");
        $fwrite(result_log, "num_images,%0d\n", nrun);
        $fwrite(result_log, "correct,%0d\n", correct);
        $fwrite(result_log, "accuracy_pct,%0d.%02d\n",
                (correct*100)/nrun, ((correct*10000)/nrun) % 100);
        $fwrite(result_log, "active_cycles,%0d\n", tot_active);
        $fwrite(result_log, "synapse_ops,%0d\n", tot_synops);
        $fwrite(result_log, "neuron_updates,%0d\n", tot_updates);
        $fwrite(result_log, "spikes_fired,%0d\n", tot_spikes);
        $fclose(result_log);

        #100;
        $finish;
    end

    initial begin
        longint unsigned timeout_ns;
        if (!$value$plusargs("timeout_ns=%d", timeout_ns))
            timeout_ns = 64'd120_000_000_000;
        /* verilator lint_off ZERODLY */
        #(timeout_ns);
        /* verilator lint_on ZERODLY */
        $display("[TIMEOUT] Simulation exceeded time limit (%0d ns)", timeout_ns);
        $finish;
    end

endmodule
