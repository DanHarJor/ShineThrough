import imas
from imas import imasdef
import scipy.io
import numpy as np

import scipy.io

def mat_to_ids(mat_in_path, ids_out_path):
    def mat_to_dict(mat_obj):
        """
        Recursively convert mat objects to nested dictionaries.
        """
        if isinstance(mat_obj, dict):
            return {k: mat_to_dict(v) for k, v in mat_obj.items()}
        elif isinstance(mat_obj, scipy.io.matlab.mat_struct):
            return {k: mat_to_dict(v) for k, v in mat_obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(mat_obj, np.ndarray):
            return [mat_to_dict(item) for item in mat_obj]
        else:
            return mat_obj

    print('LOADING .mat')
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_in_path, struct_as_record=False, squeeze_me=True)
    print('FINISHED LOADING .mat')

    # Convert to nested dictionary
    nested_dict = mat_to_dict(mat_data)
    # Now you can access your nested dictionary

    ids_s = list(nested_dict['exported_data'].keys())
    #ids_s = ['pulse_schedule', 'dataset_description', 'dataset_fair', 'summary', 'core_profiles', 'core_transport', 'core_sources', 'radiation', 'transport_solver_numerics', 'equilibrium']
    #ids_s = ['equilibrium']

    print('keys', nested_dict.keys())

    def get_obj(nested_dict, parent_key=''):
        obj = []
        values = []
        for key, value in nested_dict.items():
            #print('value_type',type(value))
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                obj, _ = get_obj(value, full_key)
                obj.extend(obj)
            else:
                obj.append(f'{full_key}')
                values.append(value)
        return obj, values

    imas_entry = imas.DBEntry(f"imas:hdf5?path={ids_out_path}", "w")
    error_count = 0
    for ids in ids_s:
        exec(f'{ids}_imas = imas.{ids}()')
        ids_dict = nested_dict['exported_data'][ids]
        obj_list, values = get_obj(ids_dict, f'{ids}_imas')
        for obj, value in zip(obj_list, values):
            if 'pulse_schedule' in obj:
                print('OBJ2:',obj,value)
            try:
                exec(f'{obj}=value')
            except:
                error_count += 1
                print("This object didn't exist in python:", obj)
                print("or there was some other error with exec(f'{obj}=value')") 
        if ids == 'pulse_schedule':
            exec(f'{ids}_imas.ids_properties.homogeneous_time = 0') #I have no idea if this is the correct value. I am putting something in because it is manditory
        exec(f'imas_entry.put({ids}_imas)')

    imas_entry.close()

mat_to_ids('/home/danharjor/ShineThrough/data/100102_n4_np3_E6_run126_IMAS.mat', 'my-data')


#eq_dict = {'ids_properties':{'occurrence_type':{'index':'orig index', 'description':'orig dec'}}}
#eq = imas.equilibrium()
#
#exe = get_key_paths(eq_dict, 'eq')
#print('\nEX',exe)
#for ex in exe:
#    exec(ex)
#print('INDEX ORIG',eq.ids_properties.occurrence_type.index)


#print('\neq',type(eq))
#print('\neq.ids_prop',type(eq.ids_properties))
#print('\neq.ids.occ',type(eq.ids_properties.occurrence_type))
#print('\neq.ids.occ.name',type(eq.ids_properties.occurrence_type.name))

#print('\nids prop comment',eq.ids_properties.comment)

#eq.ids_properties.occurrence_type.description = 'daniel description'
#eq.ids_properties.occurrence_type.index = 'daniel index'

#setattr(eq.ids_properties.occurrence_type, 'index', 'daniel index 2')

#print(eq.ids_properties.occurrence_type.index)

#print(dir(imas.equilibrium().ids_properties))
#print(dir(eq))
#print('type', type(eq))
""" 
for ids in ids_s:
    ids_dict = nested_dict['exported_data'][ids]
    ids_imas = getattr(imas, ids)
    print(list(ids_dict.keys()))
    setattr(ids_imas, list(ids_dict.keys())[0], list(ids_dict.values())[0])
    print(getattr(ids_imas, list(ids_dict.keys())[0]))
 """


""" print('LOADING MAT')
mat = scipy.io.loadmat('/home/danharjor/ShineThrough/100102_n4_np3_E6_run126_IMAS.mat')
print('FINISHED LOADING MAT')
 """



""" # Create the database entry
imas_entry = imas.DBEntry("imas:hdf5?path=my-data", "w")

# Create an empty pf_active IDS
ids = imas.pf_active()
print(type(ids))
# Set the mandatory ids_properties.homogeneous_time field
ids.ids_properties.homogeneous_time = imasdef.IDS_TIME_MODE_HOMOGENEOUS
...  # Continue filling the pf_active IDS here

# Store the pf_active IDS
imas_entry.put(ids)

# Alternatively, store the pf_active IDS as occurrence 1
#imas_entry.put(ids, occurrence=1)

imas_entry.close() """