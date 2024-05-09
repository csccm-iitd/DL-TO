import numpy as np 
import torch
import sys
import torch.nn as nn
from models.CouplingBlock import CouplingBlock
from models.CouplingOneSide import CouplingOneSide
from models.Divide_data_model import divide_data
from models.Downsample_model import Downsample
from models.Permute_data_model import Permute_data
from models.Unflat_data_model import Unflat_data
from models.flat_data_model import Flat_data

class generative_network(nn.Module):
    '''
    Args:
    s_net_t_net: scale and shift network
    input_dimension: input dimension
    for corresponding multiscale blocks.
    x: Input (BXCXHXW)
    c: conditioning data
    '''
    def __init__(self, cond_size, s_net_t_net,
                input_dimension1,input_dimension12,cond_size1, permute_a1,value_dim,input_dimension1_r,
                input_dimension2,input_dimension22,cond_size2,permute_a2,s_net_t_net2,input_dimension2_r,
                input_dimension3,input_dimension32,cond_size3,s_net_t_net3,permute_a3):
        super(generative_network, self).__init__()
        self.single_side1 = CouplingOneSide(s_net_t_net, cond_size)
        self.single_side2 = CouplingOneSide(s_net_t_net, cond_size)
        self.single_side3 = CouplingOneSide(s_net_t_net, cond_size)
        self.single_side4 = CouplingOneSide(s_net_t_net, cond_size)
        self.single_side5 = CouplingOneSide(s_net_t_net, cond_size)
        self.single_side6 = CouplingOneSide(s_net_t_net, cond_size)

        self.downsample = Downsample()

        self.coupling1 = CouplingBlock(s_net_t_net, input_dimension1,input_dimension12,cond_size1)
        self.coupling2 = CouplingBlock(s_net_t_net, input_dimension1,input_dimension12,cond_size1)
        self.coupling3 = CouplingBlock(s_net_t_net, input_dimension1,input_dimension12,cond_size1)
        self.coupling4 = CouplingBlock(s_net_t_net, input_dimension1,input_dimension12,cond_size1)
        self.coupling5 = CouplingBlock(s_net_t_net, input_dimension1,input_dimension12,cond_size1)


        self.permute = Permute_data(permute_a1,0)
        self.permute_c1 = Permute_data(permute_a1,1)
        self.permute_c2 = Permute_data(permute_a1,2)
        self.permute_c3 = Permute_data(permute_a1,3)
        self.permute_c4 = Permute_data(permute_a1,4)
    

        self.unflat1 = Unflat_data(input_dimension1_r)

        self.split = divide_data(input_dimension1,value_dim)

  
        self.coupling21 = CouplingBlock(s_net_t_net2, input_dimension2,input_dimension22,cond_size2)
        self.coupling22 = CouplingBlock(s_net_t_net2, input_dimension2,input_dimension22,cond_size2)
        self.coupling23 = CouplingBlock(s_net_t_net2, input_dimension2,input_dimension22,cond_size2)
        self.coupling24 = CouplingBlock(s_net_t_net2, input_dimension2,input_dimension22,cond_size2)


        self.permute2 = Permute_data(permute_a2,0)
        self.permute2_c1 = Permute_data(permute_a2,1)
        self.permute2_c2 = Permute_data(permute_a2,2)
        self.permute2_c3 = Permute_data(permute_a2,3)


        self.split2 = divide_data(input_dimension2,[4,4])

        self.flat2 = Flat_data()


        self.unflat2 = Unflat_data(input_dimension2_r)

        self.coupling31 = CouplingBlock(s_net_t_net3, input_dimension3,input_dimension32,cond_size3)


        self.permute3 = Permute_data(permute_a3,0)


    def forward(self, x, c1,c2,c3,c4,sample_the_data=False,forward=False,jac=False):
        if forward==True:
            '''
            print(' \n \n  ')
            print('x shape = ',x.shape)
            print('c1 shape = ',c1.shape)
            print('c2 shape = ',c2.shape)
            print('c3 shape = ',c3.shape)
            print('c4 shape = ',c4.shape)
            '''
            #1-1
            '''
            out1= self.single_side1(x,c1)
            jac0 = self.single_side1.jacobian()
            print('out1 shape = ', out1.shape)
            #1-2
            out2 = self.single_side2(out1,c1)
            jac0_1 = self.single_side2.jacobian()
            print('out2 shape = ', out2.shape)
            #1-3
            out3= self.single_side3(out2,c1)
            jac0_2 = self.single_side3.jacobian()
            print('out3 shape = ', out3.shape)
            #1-4
            out4 = self.single_side4(out3,c1)
            jac0_3 = self.single_side4.jacobian()
            print('out4 shape = ', out4.shape)
            #1-5
            out5 = self.single_side5(out4,c1)
            jac0_4 = self.single_side5.jacobian()
            print('out5 shape = ', out5.shape)
            #1-6'''
            
            out6 = self.single_side6(x,c1) 
            jac0_5 = self.single_side6.jacobian()

            #print(' \n ')
            #print('out6 shape = ', out6.shape)
            #downsample
            out7 = self.downsample(out6)
            jac_glow1 =out7
            #print('out7 shape = ', out7.shape)

            #2
            
            out12 = self.coupling1(out7,c2)
            jac1 = self.coupling1.jacobian()
            out13 = self.permute(out12)
            #print('out12 shape = ', out12.shape)
            #print('out13 shape = ', out13.shape)
            
            out14 = self.coupling2(out13,c2)
            jac1_c1 = self.coupling2.jacobian()
            out15 = self.permute_c1(out14)
            #print('out14 shape = ', out14.shape)
            #print('out15 shape = ', out15.shape)
            
            '''
            out16 = self.coupling3(out15,c2)
            jac1_c2 = self.coupling3.jacobian()
            out17 = self.permute_c2(out16)
            print('out16 shape = ', out16.shape)
            print('out17 shape = ', out17.shape)

            out18 = self.coupling4(out17,c2)
            jac1_c3 = self.coupling4.jacobian()
            out19 = self.permute_c3(out18)
            print('out18 shape = ', out18.shape)
            print('out19 shape = ', out19.shape)

            out20 = self.coupling5(out19,c2)
            jac1_c4 = self.coupling5.jacobian()
            out21 = self.permute_c4(out20)
            print('out20 shape = ', out20.shape)
            print('out21 shape = ', out21.shape)
            '''

            # out22 = self.split(out21)
            out22 = self.split(out13)
            out1s = out22[0] 
            out2s = out22[1]
            #print('Type out22', out22.type)
            #print('out22 shape = ', out22.shape)
            #print('out1s shape = ', out1s.shape)
            #print('out2s shape = ', out2s.shape)

            flat_output1 = self.flat2(out2s)
            #print('flat_output1 shape = ', flat_output1.shape)

            out31 = self.downsample(out1s)
            jac_glow2 = out31
            #print('out31 shape = ', out31.shape)

            #3

            out32 = self.coupling21(out31,c3)
            jac2 = self.coupling21.jacobian()
            out33 = self.permute2(out32)
            #print('out33 shape = ', out33.shape)

            out34 = self.coupling22(out33,c3)
            jac2_c1 = self.coupling22.jacobian()
            out35 = self.permute2_c1(out34)
            #print('out35 shape = ', out35.shape)
            '''
            out36 = self.coupling23(out35,c3)
            jac2_c2 = self.coupling23.jacobian()
            out37= self.permute2_c2(out36)
            print('out37 shape = ', out37.shape)

            out38 = self.coupling24(out37,c3)
            jac2_c3 = self.coupling24.jacobian()
            out39 = self.permute2_c3(out38)
            print('out39 shape = ', out35.shape)
            '''
            out40 = self.split2(out35)
            out1s4 = out40[0] 
            out2s4 = out40[1] 
            #print('out1s4 shape = ', out1s4.shape)
            #print('out2s4 shape = ', out2s4.shape)
            
            flat_output2 = self.flat2(out2s4)        
            flat_ds2 = self.flat2(out1s4)  
            jac_glow3 =  flat_ds2
            
            #print('flat_output2 shape = ', flat_output2.shape)
            #print('flat_ds2 shape = ', flat_ds2.shape)
            #4
            out1f = self.coupling31(flat_ds2,c4)
            jac3 = self.coupling31.jacobian()
            #print(' \n ')
            #print('out1f shape = ', out1f.shape)
            

            out_all = self.permute3(out1f)
            #print('out_all shape = ', out_all.shape)
           
            final_out  = torch.cat((flat_output1,flat_output2,out_all),dim=1)
            #print('final_out shape = ', final_out.shape)
            
            # final_out = torch.sigmoid(final_out) if we want to use sigmoid use this else not required

            #jacobian
            # jac = jac0+jac1+jac2+jac3+jac0_1+jac0_2+jac0_3+jac0_4+jac0_5+jac1_c1+jac1_c2+jac1_c3+jac1_c4+jac2_c1+jac2_c2+jac2_c3
            jac = jac0_5 + jac1 + jac3 + jac1_c1 + jac2_c1 
            return final_out, jac
        else:
            '''
            print(' \n ')
            print('else ----------------------------------')
            print('x shape = ', x.shape)
            '''
            
            #unflat the 2X32X32 data
            out1 = x[:,:2048]
            out1_unflat = self.unflat1(out1)
            #print('out1 shape = ', out1.shape)
            #print('out1_unflat shape = ', out1_unflat.shape)
            
            #unflat the 4X16X16 data            
            out2 = x[:,2048:3072]
            out2_unflat = self.unflat2(out2)
            #print('out2 shape = ', out2.shape)
            #print('out2_unflat shape = ', out2_unflat.shape)
            
            # this is considered as the  input to the INN model 1024 
            out3 = x[:,3072:]
            #permute the data
            out3p = self.permute3(out3,sample_the_data=True)
            # consider the INN model FC
            out = self.coupling31(out3p,c4,sample_the_data=True)
            out3_unflat = self.unflat2(out)
            #combine the data
            combine_out2_out3 = torch.cat((out3_unflat,out2_unflat), dim=1)
            #=========================================
            #permute the data
            out_4 =  self.permute2_c3(combine_out2_out3,sample_the_data=True)

            
            out_5 = self.coupling24(out_4,c3,sample_the_data=True)    
            #==============================================    
            #=========================================
            #permute the data
            out_4 =  self.permute2_c2(out_5,sample_the_data=True)

            
            out_5 = self.coupling23(out_4,c3,sample_the_data=True)   
            #==============================================   
            #=========================================
            #permute the data
            out_4 =  self.permute2_c1(out_5,sample_the_data=True)

            
            out_5 = self.coupling22(out_4,c3,sample_the_data=True)   
            #==============================================  Here 
            #=========================================
            #permute the data
            out_4 =  self.permute2(out_5,sample_the_data=True)

            
            out_5 = self.coupling21(out_4,c3,sample_the_data=True)   
            #==============================================  Here 

            #updample to 2X32X32
            out_6 = self.downsample(out_5,sample_the_data=True)
            #combine the data with out_1 4X32X32
            combine_out6_out1 = torch.cat((out_6,out1_unflat), dim=1)
            #=============================
            #permute
            out_7 =  self.permute_c4(combine_out6_out1,sample_the_data=True)
            
            out_8 = self.coupling5(out_7,c2,sample_the_data=True) 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c3(out_8,sample_the_data=True)
 
            out_8 = self.coupling4(out_7,c2,sample_the_data=True) 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c2(out_8,sample_the_data=True)
    
            out_8 = self.coupling3(out_7,c2,sample_the_data=True) 
 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c1(out_8,sample_the_data=True)
            
            out_8 = self.coupling2(out_7,c2,sample_the_data=True)  
            #==================================
            #=============================
            #permute
            out_7 =  self.permute(out_8,sample_the_data=True)
            
            out_8 = self.coupling1(out_7,c2,sample_the_data=True)  
            #==================================
            #upsample 1X64X64
            out_9 = self.downsample(out_8,sample_the_data=True)
            out_10 = self.single_side6(out_9,c1,sample_the_data=True)
            out_10 = self.single_side5(out_10,c1,sample_the_data=True)
            out_10 = self.single_side4(out_10,c1,sample_the_data=True)
            out_10 = self.single_side3(out_10,c1,sample_the_data=True)
            out_10 = self.single_side2(out_10,c1,sample_the_data=True)
            out_10 = self.single_side1(out_10,c1,sample_the_data=True)

            return out_10
