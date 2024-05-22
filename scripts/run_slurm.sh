# before start
mkdir -p /var/spool/slurmd
mkdir -p /var/lib/slurm-llnl
mkdir -p /var/lib/slurm-llnl/slurmd
mkdir -p /var/lib/slurm-llnl/slurmctld
mkdir -p /var/run/slurm-llnl
mkdir -p /var/log/slurm-llnl

chmod -R 755 /var/spool/slurmd
chmod -R 755 /var/lib/slurm-llnl/
chmod -R 755 /var/run/slurm-llnl/
chmod -R 755 /var/log/slurm-llnl/

chown -R slurm:slurm /var/spool/slurmd
chown -R slurm:slurm /var/lib/slurm-llnl/
chown -R slurm:slurm /var/run/slurm-llnl/
chown -R slurm:slurm /var/log/slurm-llnl/


systemctl start slurmctld
systemctl start slurmd
systemctl status slurmctld -l
# systemctl enable slurmd.service
scontrol show nodes