from dateutil import rrule
import datetime

sta = datetime.datetime(2019,1,1)
sto = datetime.datetime(2019,12,31)
sats = ['S1A','S1B']
output_listing = '/home1/datawork/agrouaze/data/sentinel1/cwave/validation_quach2020/heteroskedastic2017_version_4feb2021/listing_dates_sat_to_generate_input_output_dataset.txt'
fi = open(output_listing,'w')
cpt = 0
for dd in rrule.rrule(rrule.DAILY,dtstart=sta,until=sto):
    for sat in sats:
        #fi.write('--date %s --satellite %s\n' % (dd.strftime('%Y%m%d'),sat))
        fi.write('%s %s\n'%(dd.strftime('%Y%m%d'),sat))
        cpt += 1
fi.close()
print('nb lines to treat: ',cpt)
print('output : ',output_listing)