//-----------------------------------------------------------------------------
// tb_classify_ann.sv - INT8 ANN baseline classification testbench
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "snn_config.vh"

`ifndef SNN_ANN_HIDDEN_SHIFT
`define SNN_ANN_HIDDEN_SHIFT 16
`endif

module tb_classify_ann;

    localparam int N_INPUT       = `SNN_N_INPUT;
    localparam int N_HIDDEN      = `SNN_N_HIDDEN;
    localparam int N_OUTPUT      = `SNN_N_OUTPUT;
    localparam int NUM_TEST      = `SNN_NUM_TEST;
    localparam int HIDDEN_SHIFT  = `SNN_ANN_HIDDEN_SHIFT;
    localparam int PIXEL_WIDTH   = 8;
    localparam int OUT_WIDTH     = $clog2(N_OUTPUT);

    logic clk;
    logic rst_n;
    logic start;
    logic done;
    logic [N_INPUT*PIXEL_WIDTH-1:0] image_pixels;
    logic [OUT_WIDTH-1:0] prediction;
    logic [31:0] cycle_count;
    logic [31:0] active_cycles;
    logic [31:0] total_mac_ops;

    top_ann #(
        .N_INPUT      (N_INPUT),
        .N_HIDDEN     (N_HIDDEN),
        .N_OUTPUT     (N_OUTPUT),
        .HIDDEN_SHIFT (HIDDEN_SHIFT)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .image_pixels  (image_pixels),
        .done          (done),
        .prediction    (prediction),
        .cycle_count   (cycle_count),
        .active_cycles (active_cycles),
        .total_mac_ops (total_mac_ops)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    logic [7:0] pixel_mem [0:NUM_TEST*N_INPUT-1];
    logic [3:0] label_mem [0:NUM_TEST-1];

    longint tot_active, tot_macs;
    int correct;
    integer result_log;

    task automatic reset_dut();
        rst_n <= 1'b0;
        start <= 1'b0;
        image_pixels <= '0;
        repeat(4) @(posedge clk);
        rst_n <= 1'b1;
        repeat(2) @(posedge clk);
    endtask

    task automatic load_image(input int img);
        for (int p = 0; p < N_INPUT; p++)
            image_pixels[p*PIXEL_WIDTH +: PIXEL_WIDTH] = pixel_mem[img*N_INPUT + p];
    endtask

    task automatic classify(input int img, output int pred);
        load_image(img);
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        while (!done) @(posedge clk);
        pred = int'(prediction);
        @(posedge clk);
    endtask

    int run_first, run_last, nrun;
    string tag;

    initial begin
        int pred;
        if (!$value$plusargs("first=%d", run_first)) run_first = 0;
        if (!$value$plusargs("last=%d",  run_last))  run_last  = NUM_TEST - 1;
        if (!$value$plusargs("tag=%s",   tag))       tag       = "";
        if (run_last > NUM_TEST - 1) run_last = NUM_TEST - 1;
        nrun = run_last - run_first + 1;

        $display("==========================================================");
        $display("  INT8 ANN Baseline - Digit Classification");
        $display("  %0d -> %0d -> %0d, hidden_shift=%0d",
                 N_INPUT, N_HIDDEN, N_OUTPUT, HIDDEN_SHIFT);
        $display("  images %0d..%0d", run_first, run_last);
        $display("==========================================================");

        $readmemh("ann_pixels.mem", pixel_mem);
        $readmemh("snn_labels.mem", label_mem);

        result_log = $fopen($sformatf("classify_ann%s.csv", tag), "w");
        $fwrite(result_log, "image,label,prediction,correct,active_cycles,mac_ops\n");

        reset_dut();
        correct = 0;
        tot_active = 0;
        tot_macs = 0;

        for (int img = run_first; img <= run_last; img++) begin
            classify(img, pred);
            if (pred == int'(label_mem[img])) correct++;
            tot_active += longint'(active_cycles);
            tot_macs   += longint'(total_mac_ops);
            $fwrite(result_log, "%0d,%0d,%0d,%0d,%0d,%0d\n",
                    img, label_mem[img], pred,
                    (pred == int'(label_mem[img])) ? 1 : 0,
                    active_cycles, total_mac_ops);
            if (img - run_first < 12)
                $display("  image %0d: label=%0d predict=%0d  %s",
                         img, label_mem[img], pred,
                         (pred == int'(label_mem[img])) ? "OK" : "x");
        end

        $display("\n==========================================================");
        $display("  Classification accuracy: %0d / %0d = %0d.%02d%%",
                 correct, nrun, (correct*100)/nrun,
                 ((correct*10000)/nrun) % 100);
        $display("  INT8 ANN work:");
        $display("    Active cycles:       %0d", tot_active);
        $display("    MAC ops:             %0d", tot_macs);
        $display("==========================================================\n");

        $fclose(result_log);

        result_log = $fopen($sformatf("metrics_classify_ann%s.csv", tag), "w");
        $fwrite(result_log, "metric,value\n");
        $fwrite(result_log, "design,int8_ann\n");
        $fwrite(result_log, "num_images,%0d\n", nrun);
        $fwrite(result_log, "correct,%0d\n", correct);
        $fwrite(result_log, "accuracy_pct,%0d.%02d\n",
                (correct*100)/nrun, ((correct*10000)/nrun) % 100);
        $fwrite(result_log, "active_cycles,%0d\n", tot_active);
        $fwrite(result_log, "mac_ops,%0d\n", tot_macs);
        $fwrite(result_log, "synapse_ops,%0d\n", tot_macs);
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
