# Next Steps: Extending the Baseline

This `reduced/` folder is the clean, 1-to-1 ported baseline with zero extensions. If you want to add features **after students understand this core**, here are the safe places to build:

## Safe Extensions

### 1. New Plotting Functions
- **Where**: `plotting/`
- **Pattern**: Add a new file or function in existing files
- **Example**: `plotting/custom_heatmap.py` for a specialized visualization
- **Why it's safe**: Visualization can't break physics or dynamics

### 2. New Analysis / Sensor Modes
- **Where**: `simulation/sensor_check.py` or a new `simulation/analysis.py`
- **Pattern**: Add functions that post-process or analyze existing outputs
- **Example**: RMS error calculation, stability metrics
- **Why it's safe**: No changes to integration or field computation

### 3. Modified Potential Energy Modes
- **Where**: `simulation/potential.py`
- **Pattern**: Add a new function `energy_potential_custom()` alongside the existing three
- **Example**: A hybrid static/controlled mode
- **Why it's safe**: Still grid-based, open-loop

### 4. New Geometry Presets
- **Where**: `data/Presets/` (Excel files)
- **Pattern**: Create a new `.xlsx` in Presets
- **Steps**: 
  1. Copy an existing preset
  2. Modify coil positions / currents
  3. Run the parity check to ensure fields still compute
  4. Test against sensor data
- **Why it's safe**: Data-driven, doesn't change code

### 5. Command-Line Interface Improvements
- **Where**: `main.py`
- **Pattern**: Add new menu options that call existing functions or new analysis functions
- **Example**: Option 6 for "Batch analysis" or "Parameter sweep"
- **Why it's safe**: UI is isolated; doesn't affect simulation

## Risky Extensions (Avoid Until You're Confident)

### ❌ Feedback Control
- **Why avoid**: Requires tuning, stability analysis, full re-testing
- **If you must**: Write it in a new `control/` folder in the *parent* package, not here. Test heavily before merging into baseline
- **Better**: Build from scratch with quaternion tracking and SO(3) if you go there

### ❌ Modifying Physics Kernels
- **Why avoid**: Field kernels are parity-checked; any change breaks the guarantee
- **If you must**: Create a fork (`physics/field_custom.py`), test in isolation against sensor data, then decide if it's worth the split

### ❌ Changing Dipole Model
- **Why avoid**: Single-point dipole assumption baked into potential energy calculations
- **If you must**: Use the multi-dipole model in the main package; it's more correct but slower

### ❌ Changing Attitude Representation
- **Why avoid**: Euler angles are tightly coupled to the dynamics ODE
- **If you must**: The main package's quaternion path is there; copy from `../geometry/quaternion.py` and the full dynamics

---

## Testing Checklist for Extensions

Before shipping an extension:

1. **Run the parity check** (if you touched physics):
   ```bash
   cd ..
   python notebooks/nblib/parity_check.py
   ```
   Should output: all green

2. **Test against sensor data** (if you touched field or potential):
   ```
   main.py > option 2 > "Check simulation against sensor data"
   ```
   Should pass without large residuals

3. **Verify open-loop dynamics** (if you touched anything):
   ```
   main.py > option 4 > "Simulate pod movement"
   ```
   Should produce smooth, stable trajectories

4. **Document what you changed**:
   - Add a docstring to your new function
   - Update this file to reflect the new safe extension
   - If it's a parameter or formula, leave a citation to the source (dissertation, paper, etc.)

---

## File Ownership

| Folder | Responsibility | Risk |
|--------|---|---|
| `physics/` | Numerical kernels | **CRITICAL**: Parity check must pass |
| `simulation/` | Integration and analysis | **HIGH**: Must validate against sensor data |
| `plotting/` | Visualization | **LOW**: Can add freely |
| `geometry/` | Representation | **MEDIUM**: Affects inertia, rotation, transformations |
| `data_io/` | File I/O | **LOW**: Just copying/saving data |
| `config.py` | Constants | **MEDIUM**: Changing values affects all downstream |
| `main.py` | CLI | **LOW**: UI is isolated |

---

## Quick Win: Parameter Sweep

A safe extension that students often want:

**Goal**: Run dynamics across a 2D grid of initial conditions.

**Steps**:
1. Create `simulation/parameter_sweep.py`
2. Loop over `initial_state.x0` and `initial_state.z0` (or similar)
3. Call `run_dynamics()` for each combination
4. Aggregate results (max height, final position, stability metric)
5. Visualize in `plotting/sweep_heatmap.py`

**Why it's safe**: Calls only existing, tested functions

**Estimated effort**: 2–3 hours

---

**Questions?** Check the CLAUDE.md in the parent folder for architecture notes.
