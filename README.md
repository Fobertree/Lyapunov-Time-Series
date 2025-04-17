# Lyapunov Exponents as an Indicator of Time Series

Disclosure: majority coded under AI due to being too busy and Algory presentation deadline

MinMax scale time series to [0,1] for stability of parameters

Need to think through how to conduct parameter search
- For now, grid search? (although computationally expensive)

Implementations
- Sano-Sawada
    - Only consider using for 1500+ length time series
    - Measures neighborhood Jacobian stability based on eigenvalues
        - Greater eigenvalues -> greater instability
- Recurrence quantification analysis
    - Recurrence plots (when do states in phase space return back to same state?)
    - Euclidean distances compared between future states and more 'current states'
        - Greater distance -> greater Lyapunov exponents