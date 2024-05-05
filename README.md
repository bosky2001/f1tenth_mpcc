# F1Tenth MPCC

The [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers.git) library, must be installed independantly through the following commands, 
```
git submodule init
git submodule update
cd trajectory_planning_helpers
pip install -e .
```

The MPCC algorithms use the [casadi](https://web.casadi.org/python-api/) optimistion package, which relies on the IPOPT library. Instructions to install IPOPT can be found [here]().