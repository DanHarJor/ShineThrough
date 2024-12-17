import imaspy
import imas

#help(imaspy.ids_defs.ASCII_BACKEND)

database, pulse, run, user = 'mydatabase', 42, 42, 'myuser'
data = {'mydata':42}
temp = imaspy.DBEntry(imaspy.ids_defs.HDF5_BACKEND, database, pulse, run, user)
temp.create()
temp.put(data)
temp.close()
