Hi All,

I've setup the host engelbart to use turing student home directories.
engelbart is one of our GPU enabled HPC hosts.

So, from a turing terminal you can:

```bash
ssh -Y engelbart
```

to get a command prompt on engelbart.

From here you can see the GPU with the commands nvidia-smi or nvtop
engelbart has a 16GB v100 GPU, it's our smallest GPU but it's still pretty capable.

bourbaki is another host that has student /home and a more capable 40GB A100
But bourbaki is being used all the students in another unit, so it may be busy.
I suggest using engelbart to get going. Your work will port to a bigger GPU if it's needed.

The GPU HPC hosts all run Rocky 9 (RHEL 9) operating system which is different from turing's Fedora. You will find that much software built on turing won't run on the HPC hosts, you need to build it on engelbart (or other HPC host).

If you require space for large data files, there is a /scratch partition on most HPC hosts. Use it like this:

```bash
mkdir /scratch/comp320a
```

and keep your shared files under that directory. /scratch areas are not backed up but there is 1TB free there. Your home directory only has a 40GB quota.
/scratch is also local to the system (not network filesystem) so much much faster.

You will need to manage the permissions in /scratch/comp320a so that everyone can read them.

Regarding building your software, make sure you do this on engelbart (or bourbaki). If you do it on turing, it will not work.

Another way of building software is to use apptainer which is like docker but for HPC environments.
It's probably best to get AI to help you build an apptainer sandbox, install your python packages and create a .sif file. Then use it something like this:

```bash
apptainer exec --nv myproject.sif python train_model.py
```

If you use VS Code for development, you can leverage its "Remote - SSH" extension. Connect to turing.une.edu.au first, and from there, you can configure it to "jump" to engelbart.

Let me know if you have any questions or you need more resources.

Cheers,
UNE IT.
