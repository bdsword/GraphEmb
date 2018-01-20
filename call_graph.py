import idc
import idaapi

print("Test")
cur = idc.MinEA()
end = idc.MaxEA()
path = idc.GetIdbPath().rsplit('.')[0] + '.gdl'
idc.GenCallGdl(path, 'Call Gdl', idc.CHART_GEN_GDL)
idc.Message('Gdl file has been saved to {}\n'.format(path))
idc.Exit(0)

