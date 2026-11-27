//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member. Everything is
//! in the library (so the apps can run a node in-process); this is the command line.
fn main() -> anyhow::Result<()> {
    clamor_node::main_cli()
}
