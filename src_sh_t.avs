LoadPlugin("mvtools2.dll")
#LoadPlugin("mvtools2_stdSO2.dll")
#LoadPlugin("mvtools2_oldFP_CheckMV0.dll")
#LoadPlugin("mvtools2_pint.dll")
LoadPlugin("rawsourceplus.dll")

LoadPlugin("AddGrainC.dll")

LoadPlugin("RawSourcePlus.dll")

RawSourcePlus("out8.raw", 288,288, "Y8")

ConvertToYV12(matrix="PC.709")

AddBorders(10,10,10,10,color=$101010)

cl_0d25 = SincResize(last.width, last.height, src_left=0.25, taps=20)
cl_0d5 = SincResize(last.width, last.height, src_left=0.5, taps=20)
cl_0d75 = SincResize(last.width, last.height, src_left=0.75, taps=20)

last + cl_0d25 + cl_0d5 + cl_0d75 + cl_0d5 + cl_0d25

Loop(50)

AddBorders(0,0, 1612, 772)

AddGrain(25)

#SincResize(width * 3, height * 3, taps=20)

#return last

tr = 12 # Temporal radius
super = MSuper (mt=false, chroma=true,pel=2, hpad=8, vpad=8, levels=0)
multi_vec = MAnalyse (super, search=3, searchparam=2, multi=true, blksize=8, delta=tr, chroma=true, overlap=0, mt=false, optSearchOption=0, optPredictorType=0, levels=0, scaleCSAD=0)
#multi_vec = MAnalyse (super, search=3, searchparam=2, multi=true, blksize=8, delta=tr, chroma=false, overlap=0)
MDegrainN (super, multi_vec, tr, thSAD=400, thSAD2=390, mt=false,wpow=4, UseSubShift=1)

#super = MSuper(mt=false, pel=2)
#backward_vec2 = MAnalyse(super, isb = true, delta = 2, chroma=false)
#backward_vec1 = MAnalyse(super, isb = true, delta = 1, chroma=false)
#forward_vec1 = MAnalyse(super, isb = false, delta = 1, search=3, chroma=false,optSearchOption=1, optPredictorType=0, levels=0)
#forward_vec1 = MAnalyse(super, isb = false, delta = 1, search=3, chroma=false,optSearchOption=5, optPredictorType=0, levels=1)
#forward_vec2 = MAnalyse(super, isb = false, delta = 2, chroma=false)
#MShow(super,forward_vec1, scale=4, showsad=true)
#MDegrain2(super, backward_vec1,forward_vec1,backward_vec2,forward_vec2,thSAD=400)

SincResize(width * 3, height * 3, taps=20)

#Prefetch(6)
