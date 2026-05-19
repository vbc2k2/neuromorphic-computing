# Chapter 1: Digital Logic And RTL Basics

## 1.1 Digital Hardware Works With Bits

Digital hardware represents information using bits:

```text
0 or 1
```

Multiple bits form numbers:

```text
8 bits  -> one byte
16 bits -> common small integer width
32 bits -> common counter width
```

In SystemVerilog, a vector is written like:

```systemverilog
logic [7:0] weight;
```

That means `weight` has 8 bits, indexed from bit 7 down to bit 0.

## 1.2 Combinational Logic

Combinational logic has no memory. Its output is determined only by current
inputs.

Example:

```systemverilog
assign is_equal = (a == b);
```

If `a` or `b` changes, `is_equal` changes immediately in simulation.

Common combinational operations:

- Add
- Compare
- Select with `if` or `case`
- Decode an ID
- Compute a next state

## 1.3 Sequential Logic

Sequential logic has memory. It changes only on clock edges.

Example:

```systemverilog
always_ff @(posedge clk) begin
    count <= count + 1;
end
```

Here `count` updates on the rising edge of `clk`.

This is the basic pattern for registers, counters, state machines, and stored
neuron membrane values.

## 1.4 Clock

A clock is a periodic signal:

```text
0 -> 1 -> 0 -> 1 -> ...
```

Most hardware in this project updates on the positive edge:

```systemverilog
@(posedge clk)
```

The testbenches create a clock using:

```systemverilog
initial clk = 0;
always #5 clk = ~clk;
```

That means the clock toggles every 5 ns, so the full period is 10 ns.

## 1.5 Reset

Reset puts hardware into a known initial state.

This project uses active-low reset in many modules:

```systemverilog
if (!rst_n) begin
    state <= IDLE;
    counter <= 0;
end
```

`rst_n` means reset is active when the signal is 0.

## 1.6 Blocking vs Nonblocking Assignments

In clocked logic, use nonblocking assignment:

```systemverilog
count <= count + 1;
```

In combinational logic, use blocking assignment:

```systemverilog
always_comb begin
    next = current;
    if (start)
        next = BUSY;
end
```

This is a standard RTL style:

- `<=` in `always_ff`
- `=` in `always_comb`

## 1.7 Finite-State Machines

A finite-state machine, or FSM, is hardware that moves through named states.

Example states for a simple worker:

```text
IDLE -> READ -> PROCESS -> DONE -> IDLE
```

In RTL, an FSM usually has:

- A state register.
- Next-state combinational logic.
- Datapath actions for each state.

This project uses FSMs heavily:

- `spike_router.sv` walks one source neuron's fan-out list.
- `top.sv` controls one SNN timestep.
- `top_dense.sv` scans all sources and synapses.

## 1.8 Memories

Hardware memories are arrays of registers or inferred RAMs.

Example:

```systemverilog
logic [7:0] mem [0:255];
```

This means 256 entries, each 8 bits wide.

This project loads memory contents from files:

```systemverilog
$readmemh("csr_weight.mem", csr_weight);
$readmemb("snn_spikes.mem", spike_mem);
```

Use:

- `$readmemh` for hexadecimal text files.
- `$readmemb` for binary text files.

## 1.9 Width Matters

Hardware does not have infinite-size integers unless you ask for them.

This matters a lot for generated `.mem` files. If a file contains:

```text
000100
```

that is a 24-bit hex word. If the RTL memory is only 8 bits wide, some
simulators may reject the load or truncate it.

That is why `synapse_csr.sv` uses fixed file-load widths for CSR files:

```text
csr_dst.mem    -> 16-bit words
csr_start.mem  -> 24-bit words
csr_count.mem  -> 24-bit words
csr_weight.mem -> 8-bit words
```

The internal outputs can still be narrower. The memory file load width just
needs to match the exported file format.

## 1.10 Testbenches

A testbench is simulation-only code that drives the design.

In this project, testbenches:

- Load generated memories.
- Reset the design.
- Inject input spikes.
- Run timesteps.
- Count output spikes.
- Write CSV results.

The testbench is not the accelerator. It is the simulated environment around
the accelerator.

## 1.11 RTL vs Software Thinking

Software usually executes one instruction after another.

RTL describes hardware that exists all at once. Many things happen in parallel
on every clock edge.

For example, this project instantiates one `neuron_core` per neuron:

```text
1051 neurons -> 1051 neuron_core instances
```

In simulation that may be heavy, but architecturally it means the neuron array
is parallel.

## 1.12 Exercise

Answer these before moving on:

1. What is the difference between combinational and sequential logic?
2. Why does a hardware memory need a fixed bit width?
3. Why is reset important before classifying each image?
4. What does an FSM do?
5. What is the difference between the DUT and a testbench?

