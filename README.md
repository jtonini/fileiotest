# fileiotest

Here is an explanation of what this thing does.

```bash
./fileiotest.sh {numfiles} {user@ip.ad.dre.ss} {reportfile}
```

[1] Checks for requirements: 
  - `pv` and `vmtouch` must be present. Prints a message suggesting the location of `vmtouch` if it is not found.
  - Tries `python3` then `python3.9`. If neither works, it tells you it cannot run.

[2] Removes `reportfile` if it exists (using `unlink` in case `rm` is aliased)

[3] Creates `numfiles` of 10MB of random data.

[4] Transfers the files five times.
  - send files from local disk to `/dev/null` on the remote machine.
  - send files from local memory to `/dev/null` three times, with waits of 60 seconds between.
  - send files from local memory to a queued write on the remote machine.

[5] Clean up the garbage.
