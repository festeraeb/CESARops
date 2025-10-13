// High-performance drift calculation kernel in Rust
// src/lib.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use rayon::prelude::*;
use nalgebra::{Vector2, Vector3};
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct DriftPoint {
    pub timestamp: f64,
    pub lat: f64,
    pub lon: f64,
    pub velocity_u: f64,
    pub velocity_v: f64,
}

#[derive(Debug, Clone)]
pub struct EnvironmentalConditions {
    pub wind_speed: f64,
    pub wind_direction: f64,
    pub current_u: f64,
    pub current_v: f64,
    pub wave_height: f64,
    pub water_temp: f64,
    pub pressure: f64,
}

#[pyclass]
pub struct HighPerformanceDriftAnalyzer {
    drift_points: Vec<DriftPoint>,
    environmental_data: Vec<EnvironmentalConditions>,
}

#[pymethods]
impl HighPerformanceDriftAnalyzer {
    #[new]
    pub fn new() -> Self {
        Self {
            drift_points: Vec::new(),
            environmental_data: Vec::new(),
        }
    }

    /// Add drift track data (vectorized operation)
    pub fn add_drift_tracks(&mut self, 
        timestamps: &PyArray1<f64>,
        lats: &PyArray1<f64>,
        lons: &PyArray1<f64>,
        velocities_u: &PyArray1<f64>,
        velocities_v: &PyArray1<f64>
    ) {
        let timestamps = timestamps.readonly();
        let lats = lats.readonly();
        let lons = lons.readonly();
        let velocities_u = velocities_u.readonly();
        let velocities_v = velocities_v.readonly();

        self.drift_points = (0..timestamps.len())
            .into_par_iter()
            .map(|i| DriftPoint {
                timestamp: *timestamps.get(i).unwrap(),
                lat: *lats.get(i).unwrap(),
                lon: *lons.get(i).unwrap(),
                velocity_u: *velocities_u.get(i).unwrap(),
                velocity_v: *velocities_v.get(i).unwrap(),
            })
            .collect();
    }

    /// Add environmental conditions (vectorized operation)
    pub fn add_environmental_data(&mut self,
        wind_speeds: &PyArray1<f64>,
        wind_directions: &PyArray1<f64>,
        currents_u: &PyArray1<f64>,
        currents_v: &PyArray1<f64>,
        wave_heights: &PyArray1<f64>,
        water_temps: &PyArray1<f64>,
        pressures: &PyArray1<f64>
    ) {
        let wind_speeds = wind_speeds.readonly();
        let wind_directions = wind_directions.readonly();
        let currents_u = currents_u.readonly();
        let currents_v = currents_v.readonly();
        let wave_heights = wave_heights.readonly();
        let water_temps = water_temps.readonly();
        let pressures = pressures.readonly();

        self.environmental_data = (0..wind_speeds.len())
            .into_par_iter()
            .map(|i| EnvironmentalConditions {
                wind_speed: *wind_speeds.get(i).unwrap(),
                wind_direction: *wind_directions.get(i).unwrap(),
                current_u: *currents_u.get(i).unwrap(),
                current_v: *currents_v.get(i).unwrap(),
                wave_height: *wave_heights.get(i).unwrap(),
                water_temp: *water_temps.get(i).unwrap(),
                pressure: *pressures.get(i).unwrap(),
            })
            .collect();
    }

    /// High-performance drift prediction using physics + ML corrections
    pub fn predict_drift_trajectory(&self, 
        start_lat: f64, 
        start_lon: f64, 
        duration_hours: f64,
        time_step_minutes: f64,
        py: Python
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        
        let num_steps = (duration_hours * 60.0 / time_step_minutes) as usize;
        let dt = time_step_minutes * 60.0; // Convert to seconds
        
        let mut trajectory_times = Vec::with_capacity(num_steps);
        let mut trajectory_lats = Vec::with_capacity(num_steps);
        let mut trajectory_lons = Vec::with_capacity(num_steps);
        
        let mut current_lat = start_lat;
        let mut current_lon = start_lon;
        let mut current_time = 0.0;
        
        for step in 0..num_steps {
            trajectory_times.push(current_time);
            trajectory_lats.push(current_lat);
            trajectory_lons.push(current_lon);
            
            // Get environmental conditions for current time/location
            let env = self.interpolate_environmental_conditions(current_lat, current_lon, current_time);
            
            // Calculate physics-based drift
            let (base_u, base_v) = self.calculate_physics_drift(&env);
            
            // Apply ML corrections if available
            let (corrected_u, corrected_v) = self.apply_ml_corrections(&env, base_u, base_v);
            
            // Update position using corrected velocities
            let (delta_lat, delta_lon) = self.velocity_to_displacement(
                corrected_u, corrected_v, current_lat, dt
            );
            
            current_lat += delta_lat;
            current_lon += delta_lon;
            current_time += dt;
        }
        
        // Convert to numpy arrays
        let times_array = PyArray1::from_vec(py, trajectory_times);
        let lats_array = PyArray1::from_vec(py, trajectory_lats);
        let lons_array = PyArray1::from_vec(py, trajectory_lons);
        
        Ok((times_array.into(), lats_array.into(), lons_array.into()))
    }

    /// Ultra-fast batch drift analysis for multiple scenarios
    pub fn batch_drift_analysis(&self,
        start_positions: &PyArray2<f64>, // [N x 2] array of [lat, lon]
        object_types: Vec<String>,
        py: Python
    ) -> PyResult<Py<PyArray2<f64>>> {
        
        let start_positions = start_positions.readonly();
        let n_scenarios = start_positions.shape()[0];
        
        // Parallel processing of multiple drift scenarios
        let results: Vec<Vec<f64>> = (0..n_scenarios)
            .into_par_iter()
            .map(|i| {
                let start_lat = *start_positions.get([i, 0]).unwrap();
                let start_lon = *start_positions.get([i, 1]).unwrap();
                let object_type = &object_types[i];
                
                // Object-specific drift characteristics
                let (windage, leeway) = self.get_object_characteristics(object_type);
                
                // Run optimized drift calculation
                self.fast_drift_calculation(start_lat, start_lon, windage, leeway)
            })
            .collect();
        
        // Convert results to 2D numpy array
        let flat_results: Vec<f64> = results.into_iter().flatten().collect();
        let result_array = PyArray2::from_vec2(py, &[flat_results]).unwrap();
        
        Ok(result_array.into())
    }

    /// Analyze drift probability zones with uncertainty quantification
    pub fn calculate_probability_zones(&self,
        start_lat: f64,
        start_lon: f64,
        search_time_hours: f64,
        confidence_level: f64,
        py: Python
    ) -> PyResult<Py<PyArray2<f64>>> {
        
        // Monte Carlo simulation with environmental uncertainty
        let n_simulations = 1000;
        let mut all_endpoints: Vec<(f64, f64)> = Vec::with_capacity(n_simulations);
        
        // Parallel Monte Carlo simulations
        let endpoints: Vec<(f64, f64)> = (0..n_simulations)
            .into_par_iter()
            .map(|_| {
                // Add environmental uncertainty
                let perturbed_env = self.add_environmental_uncertainty();
                
                // Run drift simulation with uncertainty
                self.monte_carlo_drift_step(start_lat, start_lon, search_time_hours, &perturbed_env)
            })
            .collect();
        
        // Calculate probability density
        let probability_grid = self.calculate_probability_density(&endpoints, confidence_level);
        
        // Convert to numpy array
        let grid_array = PyArray2::from_vec2(py, &probability_grid).unwrap();
        Ok(grid_array.into())
    }
}

impl HighPerformanceDriftAnalyzer {
    /// Physics-based drift calculation with Great Lakes specifics
    fn calculate_physics_drift(&self, env: &EnvironmentalConditions) -> (f64, f64) {
        // Wind components
        let wind_rad = env.wind_direction.to_radians();
        let wind_u = env.wind_speed * wind_rad.sin();
        let wind_v = env.wind_speed * wind_rad.cos();
        
        // Great Lakes specific coefficients
        let windage_coeff = 0.03; // Typical for surface objects
        let stokes_coeff = 0.016 * env.wave_height.powf(0.5); // Wave-driven drift
        
        // Combined drift velocity
        let drift_u = env.current_u + (windage_coeff * wind_u) + (stokes_coeff * wind_u);
        let drift_v = env.current_v + (windage_coeff * wind_v) + (stokes_coeff * wind_v);
        
        (drift_u, drift_v)
    }
    
    /// Apply ML corrections to physics-based predictions
    fn apply_ml_corrections(&self, env: &EnvironmentalConditions, base_u: f64, base_v: f64) -> (f64, f64) {
        // Simplified ML correction - in practice would load trained models
        let correction_factor = 1.0 + 0.1 * (env.pressure - 1013.0) / 20.0;
        
        (base_u * correction_factor, base_v * correction_factor)
    }
    
    /// Convert velocity to lat/lon displacement
    fn velocity_to_displacement(&self, vel_u: f64, vel_v: f64, lat: f64, dt: f64) -> (f64, f64) {
        let meters_per_degree_lat = 111320.0;
        let meters_per_degree_lon = 111320.0 * lat.to_radians().cos();
        
        let delta_lat = (vel_v * dt) / meters_per_degree_lat;
        let delta_lon = (vel_u * dt) / meters_per_degree_lon;
        
        (delta_lat, delta_lon)
    }
    
    /// Interpolate environmental conditions for given location/time
    fn interpolate_environmental_conditions(&self, lat: f64, lon: f64, time: f64) -> EnvironmentalConditions {
        // Simplified interpolation - in practice would use spatial-temporal interpolation
        if let Some(env) = self.environmental_data.first() {
            env.clone()
        } else {
            // Default Great Lakes conditions
            EnvironmentalConditions {
                wind_speed: 5.0,
                wind_direction: 270.0,
                current_u: 0.05,
                current_v: 0.02,
                wave_height: 0.5,
                water_temp: 12.0,
                pressure: 1013.25,
            }
        }
    }
    
    /// Get object-specific drift characteristics
    fn get_object_characteristics(&self, object_type: &str) -> (f64, f64) {
        match object_type.to_lowercase().as_str() {
            "person" => (0.03, 0.1),           // High windage, moderate leeway
            "fender" | "boat_fender" => (0.08, 0.15),  // High windage, high leeway
            "life_ring" => (0.05, 0.12),      // Moderate windage, moderate leeway
            "debris" => (0.02, 0.05),         // Low windage, low leeway
            "boat" => (0.02, 0.05),           // Low windage (mostly submerged)
            _ => (0.03, 0.1),                  // Default values
        }
    }
    
    /// Fast drift calculation for batch processing
    fn fast_drift_calculation(&self, start_lat: f64, start_lon: f64, windage: f64, leeway: f64) -> Vec<f64> {
        // Simplified fast calculation
        let env = self.interpolate_environmental_conditions(start_lat, start_lon, 0.0);
        let (base_u, base_v) = self.calculate_physics_drift(&env);
        
        // Apply object-specific modifications
        let modified_u = base_u * (1.0 + windage);
        let modified_v = base_v * (1.0 + leeway);
        
        // Return end position after standard drift time
        let duration = 24.0 * 3600.0; // 24 hours in seconds
        let (delta_lat, delta_lon) = self.velocity_to_displacement(modified_u, modified_v, start_lat, duration);
        
        vec![start_lat + delta_lat, start_lon + delta_lon, modified_u, modified_v]
    }
    
    /// Monte Carlo drift simulation step
    fn monte_carlo_drift_step(&self, start_lat: f64, start_lon: f64, duration_hours: f64, env: &EnvironmentalConditions) -> (f64, f64) {
        let (drift_u, drift_v) = self.calculate_physics_drift(env);
        let duration_seconds = duration_hours * 3600.0;
        let (delta_lat, delta_lon) = self.velocity_to_displacement(drift_u, drift_v, start_lat, duration_seconds);
        
        (start_lat + delta_lat, start_lon + delta_lon)
    }
    
    /// Add environmental uncertainty for Monte Carlo
    fn add_environmental_uncertainty(&self) -> EnvironmentalConditions {
        // Add random perturbations to environmental conditions
        let base_env = self.interpolate_environmental_conditions(43.0, -87.0, 0.0);
        
        EnvironmentalConditions {
            wind_speed: base_env.wind_speed * (1.0 + 0.2 * (rand::random::<f64>() - 0.5)),
            wind_direction: base_env.wind_direction + 30.0 * (rand::random::<f64>() - 0.5),
            current_u: base_env.current_u * (1.0 + 0.3 * (rand::random::<f64>() - 0.5)),
            current_v: base_env.current_v * (1.0 + 0.3 * (rand::random::<f64>() - 0.5)),
            wave_height: base_env.wave_height * (1.0 + 0.4 * (rand::random::<f64>() - 0.5)),
            water_temp: base_env.water_temp,
            pressure: base_env.pressure + 10.0 * (rand::random::<f64>() - 0.5),
        }
    }
    
    /// Calculate probability density from Monte Carlo results
    fn calculate_probability_density(&self, endpoints: &[(f64, f64)], confidence_level: f64) -> Vec<Vec<f64>> {
        // Create probability grid (simplified implementation)
        let grid_size = 50;
        let mut grid = vec![vec![0.0; grid_size]; grid_size];
        
        // Find bounds
        let min_lat = endpoints.iter().map(|(lat, _)| *lat).fold(f64::INFINITY, f64::min);
        let max_lat = endpoints.iter().map(|(lat, _)| *lat).fold(f64::NEG_INFINITY, f64::max);
        let min_lon = endpoints.iter().map(|(_, lon)| *lon).fold(f64::INFINITY, f64::min);
        let max_lon = endpoints.iter().map(|(_, lon)| *lon).fold(f64::NEG_INFINITY, f64::max);
        
        // Populate grid with endpoint density
        for (lat, lon) in endpoints {
            let i = ((lat - min_lat) / (max_lat - min_lat) * (grid_size - 1) as f64) as usize;
            let j = ((lon - min_lon) / (max_lon - min_lon) * (grid_size - 1) as f64) as usize;
            
            if i < grid_size && j < grid_size {
                grid[i][j] += 1.0;
            }
        }
        
        // Normalize to probabilities
        let total_points = endpoints.len() as f64;
        for row in &mut grid {
            for cell in row {
                *cell /= total_points;
            }
        }
        
        grid
    }
}

/// Python module definition
#[pymodule]
fn cesarops_core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HighPerformanceDriftAnalyzer>()?;
    Ok(())
}