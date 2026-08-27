# RISC-V TACIT Decoder

## What is TACIT?

TACIT is the **timestamp annotated core instruction trace** format that is simple, efficient, and profiler-friendly.
It is also aliased as "L-trace" under some cases, to match other risc-v trace standard naming conventions.
Comparing to other trace formats that focus on compression efficiency, it tries to capture timestamps for all control flow changes.
This includes precise timing of taken and non-taken branches, inferable and uninferable jumps.  
TACIT provides rich timing information to profilers while trying to be simple and efficient in encoding.

### JSON configuration

We now support specifying a config via json. This simplifies the work as the CLI arguments has grown significantly over time.

```bash
cargo run --bin tacit-decoder --  --config configs/default.json
```

### CLI Arguments

CLI arguments may be used to overwrite the configuration specified by a json file, useful for quick iteration.

* `--dump-effective-config` - dump the static configuration after merging file and CLI. Useful for argument debugging.  
* `--header-only` - read the header packet, dump the trace runtime configuration, and exit.
* `--to-txt` - attach an analysis endpoint to dump all trace events and instructions decoded to a text file for reading
* `--to-stack-txt` - attach an analysis endpoint to dump decoded stack trace traversal to a text file for reading
* `--to-atomics` - attach an analysis endpoint to dump a trace of atomic operators and their stack frames to a text file for reading
* `--to-json` - attach an analysis endpoint to dump all trace events to a json file
* `--to-afdo` - attach an analysis endpoint to convert traces to branch counts and range counts, for afdo tools to consume
* `--to-gcda` - attach an analysis endpoint to convert traces to a .gcda file. Needs to speicify the source gcno file
  * `--gcno [path/to/.gcno]` - specify the path to the .gcno file for the gcda endpoint to use
* `--to-speedscope` - attach an analysis endpoint to convert traces to speedscope json format for stack frame visualization
* `--to-vpp` - attach an analysis endpoint to analyze traces for the path variation time for identifying optimization opportunities

### Adding Your Own Analysis Endpoint

TACIT decoder is designed with effortless integration of new analysis endpoints.
To do this, the user need to:

1. Add an argument to enable the analysis in `src/main.rs`.
2. Implement the interface in `backend/abstract_receiver.rs`, including:
   1. `_bump_checksum`. This can be however the analysis wish to check for the integrity of the generated alaysis.
   2. `_receive_entry`. This is what the analyzer should behave upon each new trace event.
   3. `_flush`. This is the behavior of the analyzer after all events are processed.
