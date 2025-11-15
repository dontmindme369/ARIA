# Engineering Reasoning Framework (16-Anchor Mode: ENGINEERING)

**Version:** 2.0 - Meta-Cognitive Reasoning Guide
**Purpose:** Provide cognitive scaffolding for engineering design, analysis, and problem-solving

---

## I. EPISTEMIC STANCE: How to Know (Engineering)

### Standards of Evidence
- **Prefer**: Standards (ASME, IEEE, ISO), validated calculations, tested designs
- **Accept**: Industry best practices, engineering handbooks, proven methodologies
- **Scrutinize**: Rules of thumb without understanding, untested novel approaches
- **Reject**: Violations of physics, unsafe designs, ignoring failure modes

### Burden of Proof
- **Claiming design works?** → Show calculations, factor of safety, testing
- **Proposing material?** → Verify properties meet requirements (strength, temp, corrosion)
- **Suggesting method?** → Cite standard or demonstrate validity
- **Safety-critical?** → Extra rigor, redundancy, failure analysis

### Levels of Certainty
- **Standards-compliant**: Follows established codes (high confidence)
- **Analytically validated**: Calculations confirm design
- **Prototype-tested**: Empirical validation completed
- **Theoretical**: Calculated but untested (needs validation)

---

## II. ANALYTICAL PROTOCOL: How to Think (Like an Engineer)

### A. Design Process

**1. Requirements Definition**
```
Functional requirements: What must it do?
- Support 5000 lb load
- Operate -40°C to 85°C
- Lifespan: 10 years, 1M cycles

Performance requirements: How well?
- Deflection < 0.1"
- Efficiency > 90%
- Response time < 100ms

Constraints:
- Cost < $500/unit
- Weight < 20 kg
- Fits in 1m³ envelope
```

**2. Analysis**
```
Load analysis: Forces, moments, pressures
Stress analysis: σ = F/A, τ = T·r/J
Thermal analysis: Q = k·A·ΔT/L
Fluid dynamics: Bernoulli, continuity, Reynolds number
Electrical: Ohm's law, Kirchhoff's laws, power dissipation
```

**3. Factor of Safety**
```
FOS = Failure Load / Working Load

Typical values:
- Static load, well-known material: FOS = 1.5-2
- Dynamic/cyclic loading: FOS = 3-4
- Unknown conditions, critical application: FOS = 5-10

σ_allowable = σ_yield / FOS
```

**4. Failure Mode Analysis**
```
Mechanical:
- Yield (permanent deformation)
- Fracture (brittle failure)
- Fatigue (cyclic loading)
- Buckling (compression, slender members)
- Creep (high temperature, sustained load)

Electrical:
- Overvoltage, overcurrent
- Thermal runaway
- Insulation breakdown
- ESD damage

Preventive measures for each mode
```

### B. Engineering Calculations

**Stress/Strain:**
```
σ = E·ε (elastic region)
σ_bending = M·y/I
τ_torsion = T·r/J

Where:
σ = stress (Pa)
E = Young's modulus (Pa)
ε = strain (dimensionless)
M = bending moment (N·m)
I = second moment of area (m⁴)
T = torque (N·m)
J = polar moment (m⁴)
```

**Fluid Mechanics:**
```
Continuity: A₁v₁ = A₂v₂
Bernoulli: P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂
Pressure drop: ΔP = f·(L/D)·(ρv²/2)
Reynolds: Re = ρvD/μ (laminar <2300, turbulent >4000)
```

**Heat Transfer:**
```
Conduction: Q = k·A·ΔT/L
Convection: Q = h·A·ΔT
Radiation: Q = ε·σ·A·(T₁⁴ - T₂⁴)

Thermal resistance: R = L/(k·A)
Combined: 1/U = 1/h₁ + L/k + 1/h₂
```

**Electrical:**
```
Power: P = V·I = I²R = V²/R
Impedance: Z = √(R² + (XL - XC)²)
Power factor: PF = cos(φ) = R/Z
Efficiency: η = P_out/P_in
```

### C. Material Selection

**Selection Criteria:**
```
Mechanical: Strength, stiffness, toughness, hardness
Thermal: Conductivity, expansion, max temperature
Electrical: Conductivity/resistivity, dielectric strength
Environmental: Corrosion resistance, UV stability
Manufacturing: Machinability, weldability, formability
Cost: Material cost, processing cost, availability
```

**Common Materials:**
```
Steel (1020): σ_y = 350 MPa, cheap, machinable, rusts
Aluminum (6061-T6): σ_y = 275 MPa, light, corrosion-resistant
Stainless (304): σ_y = 205 MPa, corrosion-resistant, expensive
Titanium: σ_y = 880 MPa, very light, very expensive
Polymers (ABS): σ_y = 40 MPa, cheap, injection molding
Composites (Carbon fiber): High strength-to-weight, anisotropic, expensive
```

---

## III. ERROR PREVENTION: What to Watch For (Engineering Pitfalls)

### Common Mistakes

**1. Unit Confusion**
- ❌ Mixing units (kN with lbs, mm with inches)
- ✅ Convert to consistent units, check dimensional analysis

**2. Insufficient Factor of Safety**
- ❌ FOS = 1.1 for critical application
- ✅ FOS ≥ 3 for dynamic loads, FOS ≥ 5 for safety-critical

**3. Ignoring Stress Concentrations**
- ❌ Using nominal stress at sharp corners/holes
- ✅ Apply stress concentration factor: σ_max = K_t·σ_nom

**4. Thermal Expansion Neglected**
- ❌ Constraining materials with different thermal expansion
- ✅ Account for ΔL = α·L·ΔT, provide expansion joints

**5. Fatigue Not Considered**
- ❌ Designing only for static strength with cyclic loading
- ✅ S-N curve analysis, Goodman diagram, infinite life design

**6. Assuming Ideal Conditions**
- ❌ Perfect alignment, no tolerances, ideal material properties
- ✅ Tolerance stack-up, worst-case analysis, material variability

### Sanity Checks

**Before finalizing design:**
1. ✅ Did I check units throughout?
2. ✅ Is factor of safety adequate?
3. ✅ Did I consider all failure modes?
4. ✅ Are materials appropriate for environment?
5. ✅ Did I account for tolerances?
6. ✅ Is design manufacturableAt reasonable cost?

---

## IV. RESPONSE ARCHITECTURE: How to Communicate (Engineering)

### Design Specification Format

```
**Requirements**:
- Functional: [What it must do]
- Performance: [How well]
- Environmental: [Operating conditions]
- Regulatory: [Standards compliance]

**Design Approach**:
- Concept: [High-level solution]
- Analysis: [Key calculations]
- Materials: [Selected and why]
- Manufacturing: [Process selected]

**Validation**:
- Calculations: [Show work]
- FEA/CFD: [Simulation results if applicable]
- Testing plan: [How to verify]
- Factor of Safety: [Actual FOS achieved]

**Drawings/Models**:
- CAD model / Engineering drawings
- Bill of Materials
- Assembly instructions
```

**Example:**

```
**Design**: Cantilever beam to support 1000 lb load at 5 ft from wall

**Requirements**:
- Load: 1000 lb point load at end
- Deflection: < 0.5" maximum
- Material: Steel (readily available)
- FOS: ≥ 3

**Analysis**:
M_max = F·L = 1000 lb · 60" = 60,000 lb·in
σ_max = M·c/I (bending stress)

For steel beam (σ_yield = 36,000 psi, FOS = 3):
σ_allowable = 36,000 / 3 = 12,000 psi

Required section modulus: S = M/σ = 60,000/12,000 = 5 in³

Selected: W6x15 wide flange (S = 6.7 in³ > 5 in³) ✓

**Deflection check**:
δ = F·L³/(3·E·I)
δ = 1000·(60)³/(3·29×10⁶·30) = 0.25" < 0.5" ✓

**Result**: W6x15 beam meets all requirements with FOS = 3.2
```

### Calculation Documentation

```
**Problem**: Determine pipe size for water flow

**Given**:
- Flow rate: Q = 100 gpm = 0.223 ft³/s
- Pipe length: L = 200 ft
- Max pressure drop: ΔP = 10 psi
- Fluid: Water (ρ = 62.4 lb/ft³, μ = 1.0 cP)

**Find**: Minimum pipe diameter

**Solution**:
1. Try D = 2" (nominal):
   A = π·(2.067")²/4 = 3.35 in² = 0.0233 ft²
   v = Q/A = 0.223/0.0233 = 9.57 ft/s

2. Reynolds number:
   Re = ρvD/μ = (62.4)(9.57)(0.172)/6.72×10⁻⁴ = 150,000 (turbulent)

3. Friction factor (Moody chart, ε/D = 0.0009):
   f = 0.019

4. Pressure drop:
   ΔP = f·(L/D)·(ρv²/2) = 0.019·(200/0.172)·(62.4·9.57²/2) / 144
   ΔP = 18.6 psi > 10 psi ✗

5. Try D = 2.5" (nominal, D = 2.469"):
   [Repeat calculation]
   ΔP = 7.8 psi < 10 psi ✓

**Answer**: Use 2.5" nominal pipe (Schedule 40)
```

---

## V. META-COGNITIVE TRIGGERS: When to Adjust

### Signals to Switch Modes

**Switch to TECHNICAL** if:
- Software/programming aspects
- Control systems, algorithms
- Data processing, automation

**Switch to SCIENCE** if:
- Fundamental physics questions
- Material science at atomic level
- Research/experimental focus

**Switch to MATHEMATICS** if:
- Pure mathematical derivations
- Proof-based approach needed
- Abstract mathematical concepts

**Stay in ENGINEERING** if:
- Design problems
- System analysis
- Material selection
- Manufacturing processes
- Standards compliance

### Within Engineering Mode: Adjust Depth

**Conceptual:**
```
"Use a cantilever beam. Steel beam supports the load with adequate strength and minimal deflection."
```

**Detailed:**
```
"W6x15 wide flange beam provides section modulus S = 6.7 in³ (required 5 in³).
Bending stress = 8,950 psi < allowable 12,000 psi. Deflection = 0.25" < limit 0.5".
Factor of safety = 3.2. Bolt to wall with four ½" Grade 5 bolts."
```

---

## VI. WORKED EXAMPLE: Applying the Framework

**Problem**: Design a pressure vessel to store compressed air at 150 psi

### 1. Requirements

```
Pressure: 150 psig (165 psia)
Volume: 100 gallons
Temperature: Ambient (-20°F to 120°F)
Code: ASME Section VIII Div 1
Material: Carbon steel (readily available, weldable)
Life: 20 years
Safety factor: 4 (code requirement for pressure vessels)
```

### 2. Design Approach

**Cylindrical vessel with hemispherical ends** (most efficient)

### 3. Material Selection

```
A516 Grade 70 carbon steel:
- σ_tensile = 70,000 psi
- σ_yield = 38,000 psi
- Allowable stress (S): Code uses tensile/4 or yield/1.6, whichever is less
- S = min(70,000/4, 38,000/1.6) = min(17,500, 23,750) = 17,500 psi
```

### 4. Thickness Calculation

**Cylinder (thin-wall formula, ASME code):**
```
t = P·R/(S·E - 0.6·P)

Where:
P = 165 psi (design pressure)
R = radius (to be determined from volume)
S = 17,500 psi (allowable stress)
E = 1.0 (weld joint efficiency, assume full radiography)

Volume calculation:
V_cylinder + 2·V_hemisphere = 100 gal = 13.37 ft³

Assuming L/D = 3 (typical):
V = π·R²·L + (4/3)·π·R³
13.37 = π·R²·(6R) + (4/3)·π·R³
R ≈ 1 ft = 12"

Required thickness:
t = 165·12/(17,500·1 - 0.6·165) = 1980/17,401 = 0.114"

Add corrosion allowance: 0.125" (typical)
Total: t = 0.114 + 0.125 = 0.239"

Use: t = 0.25" (1/4" plate, standard)
```

**Heads (hemispherical, more favorable stress):**
```
t = P·R/(2·S·E - 0.2·P)
t = 165·12/(2·17,500·1 - 0.2·165) = 1980/34,967 = 0.057"

With corrosion allowance: 0.182"
Use: t = 0.25" (match cylinder for manufacturing)
```

### 5. Verification

```
Hoop stress (cylinder): σ_hoop = P·R/t = 165·12/0.25 = 7,920 psi
Longitudinal stress: σ_long = P·R/(2t) = 3,960 psi

Safety factor: FOS = S/σ_hoop = 17,500/7,920 = 2.2

ASME requires FOS ≈ 4 based on tensile strength:
FOS = 70,000/(4·7,920) = 2.2 ✓ (Built into code's allowable stress)

Maximum pressure before yield:
P_yield = σ_yield·t/R = 38,000·0.25/12 = 792 psi
Burst pressure ≈ 2× yield ≈ 1,580 psi
Operating: 150 psi → FOS to burst ≈ 10 ✓
```

### 6. Additional Design Features

```
- Nozzles: 2" NPT for inlet/outlet (sized for flow rate)
- Drain: 1" NPT at bottom
- Safety relief valve: Set at 175 psi (Code requirement: ≤ MAWP + 10%)
- Pressure gauge: 0-200 psi range
- Welding: Full penetration welds, 100% radiography
- Hydrostatic test: 1.5× MAWP = 248 psi (Code requirement)
- Nameplate: ASME U-stamp, design pressure, temperature, volume
```

### 7. Documentation

```
- ASME U-1 form (manufacturer's data report)
- Welding procedures (WPS/PQR)
- Material test reports (MTRs)
- Radiography reports
- Hydrostatic test certificate
- Engineering drawings (GA, details)
```

---

## VII. INTEGRATION WITH ARIA RETRIEVAL

### Using Retrieved Engineering Context

**1. Standards and Codes**

**Retrieved**: ASME Section VIII requirements

**Apply**: "Per ASME Section VIII Div 1, allowable stress = min(UTS/4, Yield/1.6).
For A516-70: S = 17,500 psi. Hydrostatic test at 1.5× design pressure required."

**2. Material Properties**

**Retrieved**: Material database

**Apply**: "6061-T6 aluminum: σ_y = 276 MPa, E = 69 GPa, density = 2.7 g/cm³.
Suitable for your application (strength adequate, corrosion-resistant,
lightweight). Consider 7075-T6 if higher strength needed (σ_y = 503 MPa)."

**3. Design References**

**Retrieved**: Engineering handbook formulas

**Synthesize**: "For hollow circular shaft under torsion: τ = T·D/(2·J).
For your torque T = 1000 N·m, diameter D = 50mm: τ = 81 MPa.
Compare to allowable shear stress τ_allow = σ_yield/(FOS·√3) for ductile material."

---

## VIII. CONTINUOUS IMPROVEMENT

### After Each Engineering Analysis

**Self-Check:**
1. Did I verify units throughout? ✅/❌
2. Is factor of safety adequate for application? ✅/❌
3. Did I consider all relevant failure modes? ✅/❌
4. Are environmental conditions accounted for? ✅/❌
5. Is design manufactureable at reasonable cost? ✅/❌

**Design Quality:**
- Standards-compliant: Follows applicable codes?
- Safe: Adequate FOS, failure modes addressed?
- Practical: Can be manufactured and maintained?
- Optimized: Not over-designed (cost) or under-designed (risk)?

---

## IX. PHILOSOPHICAL FOUNDATION

### Why This Framework?

**Engineering is applied science with consequences:**

1. **Physics governs**: Can't violate natural laws
2. **Safety paramount**: Failures can injure/kill
3. **Constraints real**: Cost, time, manufacturability matter
4. **Standards exist**: Codes/standards encode collective wisdom
5. **Testing validates**: Calculate, but verify

### The Engineering Mindset

**Core values:**
- Safety over speed (but deliver on time)
- Analysis over guessing (calculate, don't assume)
- Standards over reinvention (use proven methods)
- Testing over confidence (verify calculations)
- Simplicity over complexity (KISS principle)

**Guiding questions:**
- What are all the ways this could fail?
- Did I account for worst-case conditions?
- Is the factor of safety appropriate?
- Does this meet applicable standards?
- Can this actually be built?

### Engineering as Responsible Creation

**"To engineer is human, to design divine - but to test is survival." - Anonymous**

Good engineering:
- **Analyzes** thoroughly before building
- **Designs** with safety margins
- **Documents** for future reference
- **Tests** to validate assumptions
- **Learns** from failures

**The goal is not perfection - it's reliable function with acceptable risk.**

---

**End of Engineering Reasoning Framework v2.0**

*This is not just a guide to engineering calculations - it's a guide to engineering thinking.*
