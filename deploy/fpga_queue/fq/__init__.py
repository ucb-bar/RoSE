"""fq — a load-balancing queueing daemon for FireSim jobs on AWS F2 FPGAs.

See ``docs/FPGA_QUEUE_DESIGN.md`` for the architecture and the reasoning
behind the resource model.  The short version:

  * The schedulable resource is a **lane**: a uniquely-tagged FireSim run
    farm (``run_farm_tag`` / ``fsimcluster``) plus the FPGAs in it.  Lanes
    are never shared, because FireSim requests *and releases* run-farm hosts
    by tag -- two concurrent jobs on one tag will terminate each other's
    instances.
  * Exclusion is a ``flock`` per lane, held by a per-job **runner** process
    for the whole job.  The kernel releases it when the runner dies, which
    is what makes crash recovery free.
  * The daemon is the only writer of the state DB.  Clients talk to it over
    a unix socket and are identified by ``SO_PEERCRED``, so a submitter
    cannot forge its identity or edit another user's job.
"""

__version__ = "0.1.0"
