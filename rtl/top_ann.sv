`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// top_ann.sv - Sequential INT8 ANN inference baseline
//
// Description:
//   Conventional dense MLP inference baseline for v2 comparisons. This module
//   runs the same trained 784->256->10 model family as the SNN paths, but as a
//   quantized ANN:
//       raw uint8 pixels -> int8 W1 -> ReLU uint8 hidden -> int8 W2 -> argmax
//
//   It is intentionally simple: one multiply-accumulate per cycle. That makes
//   operation counts explicit and gives a clean baseline against the dense SNN
//   and event-driven SNN designs.
//-----------------------------------------------------------------------------

module top_ann #(
    parameter int N_INPUT       = 784,
    parameter int N_HIDDEN      = 256,
    parameter int N_OUTPUT      = 10,
    parameter int WEIGHT_WIDTH  = 8,
    parameter int PIXEL_WIDTH   = 8,
    parameter int ACC_WIDTH     = 48,
    parameter int HIDDEN_SHIFT  = 16,
    parameter     W1_FILE       = "ann_w1.mem",
    parameter     B1_FILE       = "ann_b1.mem",
    parameter     W2_FILE       = "ann_w2.mem",
    parameter     B2_FILE       = "ann_b2.mem"
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start,
    input  logic [N_INPUT*PIXEL_WIDTH-1:0] image_pixels,
    output logic                         done,
    output logic [$clog2(N_OUTPUT)-1:0]  prediction,
    output logic [31:0]                  cycle_count,
    output logic [31:0]                  active_cycles,
    output logic [31:0]                  total_mac_ops
);

    localparam int OUT_WIDTH = $clog2(N_OUTPUT);

    typedef enum logic [2:0] {
        A_IDLE,
        A_H_INIT,
        A_H_MAC,
        A_H_STORE,
        A_O_INIT,
        A_O_MAC,
        A_O_STORE,
        A_DONE
    } ann_state_t;

    ann_state_t state;

    logic signed [WEIGHT_WIDTH-1:0] w1 [0:N_HIDDEN*N_INPUT-1];
    logic signed [31:0]             b1 [0:N_HIDDEN-1];
    logic signed [WEIGHT_WIDTH-1:0] w2 [0:N_OUTPUT*N_HIDDEN-1];
    logic signed [31:0]             b2 [0:N_OUTPUT-1];
    logic [7:0]                     hidden_act [0:N_HIDDEN-1];

    int h_idx;
    int in_idx;
    int o_idx;
    logic signed [ACC_WIDTH-1:0] acc;
    logic signed [ACC_WIDTH-1:0] best_score;

    initial begin
        $readmemh(W1_FILE, w1);
        $readmemh(B1_FILE, b1);
        $readmemh(W2_FILE, w2);
        $readmemh(B2_FILE, b2);
    end

    function automatic logic [PIXEL_WIDTH-1:0] pixel_at(input int idx);
        pixel_at = image_pixels[idx*PIXEL_WIDTH +: PIXEL_WIDTH];
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] pixel_mac(input int idx,
                                                              input logic signed [WEIGHT_WIDTH-1:0] weight);
        pixel_mac = $signed({1'b0, pixel_at(idx)}) * weight;
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] hidden_mac(input int idx,
                                                               input logic signed [WEIGHT_WIDTH-1:0] weight);
        hidden_mac = $signed({1'b0, hidden_act[idx]}) * weight;
    endfunction

    function automatic logic [7:0] relu_quant(input logic signed [ACC_WIDTH-1:0] value);
        logic signed [ACC_WIDTH-1:0] shifted;
        begin
            if (value <= 0) begin
                relu_quant = 8'd0;
            end else begin
                shifted = value >>> HIDDEN_SHIFT;
                if (shifted > 255)
                    relu_quant = 8'hFF;
                else
                    relu_quant = shifted[7:0];
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= A_IDLE;
            done           <= 1'b0;
            prediction     <= '0;
            cycle_count    <= '0;
            active_cycles  <= '0;
            total_mac_ops  <= '0;
            h_idx          <= 0;
            in_idx         <= 0;
            o_idx          <= 0;
            acc            <= '0;
            best_score     <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (state != A_IDLE && state != A_DONE)
                active_cycles <= active_cycles + 1;

            case (state)
                A_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        cycle_count   <= '0;
                        active_cycles <= '0;
                        total_mac_ops <= '0;
                        prediction    <= '0;
                        best_score    <= '0;
                        h_idx         <= 0;
                        in_idx        <= 0;
                        o_idx         <= 0;
                        state         <= A_H_INIT;
                    end
                end

                A_H_INIT: begin
                    acc    <= b1[h_idx];
                    in_idx <= 0;
                    state  <= A_H_MAC;
                end

                A_H_MAC: begin
                    acc           <= acc + pixel_mac(in_idx, w1[h_idx*N_INPUT + in_idx]);
                    total_mac_ops <= total_mac_ops + 1;
                    if (in_idx == N_INPUT - 1)
                        state <= A_H_STORE;
                    else
                        in_idx <= in_idx + 1;
                end

                A_H_STORE: begin
                    hidden_act[h_idx] <= relu_quant(acc);
                    if (h_idx == N_HIDDEN - 1) begin
                        o_idx <= 0;
                        state <= A_O_INIT;
                    end else begin
                        h_idx <= h_idx + 1;
                        state <= A_H_INIT;
                    end
                end

                A_O_INIT: begin
                    acc    <= b2[o_idx];
                    h_idx  <= 0;
                    state  <= A_O_MAC;
                end

                A_O_MAC: begin
                    acc           <= acc + hidden_mac(h_idx, w2[o_idx*N_HIDDEN + h_idx]);
                    total_mac_ops <= total_mac_ops + 1;
                    if (h_idx == N_HIDDEN - 1)
                        state <= A_O_STORE;
                    else
                        h_idx <= h_idx + 1;
                end

                A_O_STORE: begin
                    if (o_idx == 0 || acc > best_score) begin
                        best_score <= acc;
                        prediction <= OUT_WIDTH'(o_idx);
                    end
                    if (o_idx == N_OUTPUT - 1) begin
                        done  <= 1'b1;
                        state <= A_DONE;
                    end else begin
                        o_idx <= o_idx + 1;
                        state <= A_O_INIT;
                    end
                end

                A_DONE: begin
                    if (!start)
                        state <= A_IDLE;
                end

                default: state <= A_IDLE;
            endcase
        end
    end

endmodule

