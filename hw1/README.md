# Homework 1

## Task 1: Get the CPU information of a remote server

Complete the following steps:

1. (login to remote server) Login to a remote server with `ssh`, with the following information:
   1. Username: `group1`
   2. Password: `1`
   3. Server IP address: `10.100.0.179`
2. (run task) Create a directory with your name in your home directory (the default directory after you logged in) of the remote server and then change the directory to the created one, e.g. if your name is `Zhong-Yi NI`, then the directory is `zhongyini`.
    ```bash
    mkdir zhongyini
    cd zhongyini
    ```
    Then run the following commands:
    ```bash
    lscpu > lscpu.txt  # get the CPU information and save it to the file
    date >> lscpu.txt  # add the current date and time to the file
    cat lscpu.txt  # check the content of this file
    ```
3. (download file) Now switch to your local machine and copy the `lscpu.txt` file (at the remote server) to a local directory using `scp`:
    ```bash
    scp group1@10.100.0.179:huichengzhang/lscpu.txt path/to/your/local/working/directory
    ```
4. (sample submission) As an example, your final submission should be like: https://github.com/CodingThrust/AMAT5315-2025Fall-Homeworks/pull/1

To submit your homework, please follow the instructions in the [README.md](../README.md) file.

## Task 2: Install Julia

Follow the instructions in the [Julia Setup](http://scfp.jinguo-group.science/chap1-julia/julia-setup.html) to install the Julia programming language.
For Windows users, please install Julia in windows subsystem for linux (WSL). Then you should follow the Linux instructions.

Please bring your laptop to the classroom next time.
No worries if you stuck at some steps. The instructor and the TA will help in the hands-on session.
