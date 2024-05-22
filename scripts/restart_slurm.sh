systemctl restart slurmctld
systemctl restart slurmd
systemctl status slurmctld -l
# nano /var/log/slurm-llnl/slurmd.log
# nano /var/log/slurm-llnl/slurmctld.log