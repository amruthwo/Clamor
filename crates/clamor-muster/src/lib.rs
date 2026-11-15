//! The muster: find out what each device in the swarm can do, how well the devices can talk
//! to each other, and which models the group as a whole can run, *before* any model is chosen.

pub mod bench;
pub mod net;
pub mod plan;
pub mod probe;
