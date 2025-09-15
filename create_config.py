import yaml
config = {
    'erddap': {
        'lmhofs': 'https://coastwatch.glerl.noaa.gov/erddap',
        'rtofs': 'https://coastwatch.pfeg.noaa.gov/erddap',
        'hycom': 'https://tds.hycom.org/erddap'
    },
    'drift_defaults': {
        'dt_minutes': 10,
        'duration_hours': 24,
        'windage': 0.03,
        'stokes': 0.01
    },
    'seeding': {
        'default_radius_nm': 2.0,
        'default_rate': 60
    }
}
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Config created')