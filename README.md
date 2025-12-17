# fileiotest

Here is an explanation of what this thing does.

`./randomfiles.sh {numfiles} {user@ip.ad.dre.ss}`

[1] Checks to see if `vmtouch` is present. If not, it builds it.

[2] Checks to see if `pv` is present. If not, it installs it.

[3] Creates `numfiles` of 10MB of random data. 

[4] Forces them to be read into memory with `vmtouch`. Given that they have just been created, they might be 
memory resident already. The second line in the script that runs `vmtouch` just checks to see that they
are really there. Uncomment to diagnose problems.

[5] Transfer across the wire. `pv` monitors the pipe, and prints stats. Some notes on all those options:

- `find` is used in case there are more files that match `*.iotest` than the command line can handle.
- `sort` gives a predictable/repeatable order.
- -r	current (instantaneous) rate
- -a	average rate
- -8	bits per second (not bytes)
- -t	elapsed time
- -p	progress bar
- -e	ETA
- `ssh -T` don't create a login session at the other end
- `BatchMode=yes` "get in; get out"

[6] Clean up the test files.
