#include <ida.idc>;
#include <idc.idc>;

static main()
{
    auto retval;
    auto str_gdlpath;
    str_gdlpath = GetIdbPath();
    str_gdlpath = substr(str_gdlpath, 0, strlen(str_gdlpath) - 4) + ".gdl";
    retval = GenCallGdl(str_gdlpath, "Call Gdl", CHART_NOLIBFUNCS | CHART_GEN_GDL);

    // necessary for ida to exit in batch mode
    Exit(0);
}
