# GPU access

Remote HPC GPU access is available. Several nodes with different GPUs are available.
`engelbart`, `bourbaki` and `carmack` all have access to your home directory.
`engelbart` and `carmack` require you to jump host through `turing`. `engelbart` is
the preferred node for this project as it has the least demand (recommendation from
UNE IT).

Node GPUs are:

- `engelbart` V100 16gb
- `bourbaki` A100 40gb
- `carmack` L40 48gb

## Notes from UNE IT

From a turing terminal you can `ssh -Y engelbart` to get a command prompt on
engelbart. From there, `nvidia-smi` or `nvtop` show the GPU.

The GPU HPC hosts all run Rocky 9 (RHEL 9), which is different from turing's Fedora.
Much software built on turing won't run on the HPC hosts; you need to build it on
engelbart (or another HPC host).

If you require space for large data files, there is a `/scratch` partition on most
HPC hosts. Use it like this:

```bash
mkdir /scratch/comp320a
```

and keep your shared files under that directory. `/scratch` areas are not backed up
but there is 1TB free there. Your home directory only has a 40GB quota. `/scratch`
is also local to the system (not a network filesystem) so it is much faster. You
will need to manage the permissions in `/scratch/comp320a` so that everyone can read
them.

Build your software on engelbart (or bourbaki). If you build it on turing, it will
not work.

Another way of building software is to use apptainer, which is like docker but for
HPC environments. Build an apptainer sandbox, install your python packages and create
a `.sif` file, then use it something like this:

```bash
apptainer exec --nv myproject.sif python train_model.py
```

If you use VS Code for development, you can use its "Remote - SSH" extension. Connect
to `turing.une.edu.au` first, and from there configure it to "jump" to engelbart.
