'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 10000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.4.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = OQT-TC01  OQT-TC01\OQT_TC01
'<Header End>
#Include ADwinPro_All.Inc
#Define DIO_MODULE 1 
#Define AI_MODULE 2
#Define AO_MODULE 3

Dim old_time, new_time, time_diff as long
Dim ao_index as long

Dim DATA_30[1022] AS LONG '511 Pair FIFO Array [...,time,pattern]
Dim DATA_50[40000] AS LONG 'Total 10000 points for 8 DACs(AO) (4(32bit)*10000, each 16bit for DAC)

Dim Data_40[10000], Data_41[10000] As Long
Dim Data_42[10000], Data_43[10000] As Long
Dim Data_44[10000], Data_45[10000] As Long
Dim Data_46[10000], Data_47[10000] As Long

Init:
  'Turn on LED of the Modules
  P2_Set_LED(DIO_MODULE, 1)
  P2_Set_LED(AI_MODULE, 1)
  P2_Set_LED(AO_MODULE, 1)
  'Digital Channel Setting
  P2_DigProg(DIO_MODULE, Par_1) 'Set Digital Input / Output Ch
  P2_Dig_FIFO_Mode(DIO_MODULE, 3)'Set DIO FIFO as Relative output
  P2_Digout_FIFO_Clear(DIO_MODULE) 'Clear FIFO 
  P2_Digout_FIFO_Enable(DIO_MODULE, Par_31) 'Make Selected Output CH as FIFO
  P2_Digout_Fifo_Write(DIO_MODULE, Par_32, DATA_30, Par_33)'Write FIFO Pattern
  P2_Digout_FIFO_Start(Shift_Left(1, DIO_MODULE - 1)) 'Start Digital Output FIFO '10'
  
  old_time = Read_Timer()
  ao_index = 1
  
  'Analog Input Channer Setting
  P2_Set_Average_Filter(AI_MODULE, Par_40)
  P2_Burst_Init(AI_MODULE, Par_41, 0, Par_42, Par_43, Par_44)
  P2_Burst_Start(Shift_Left(1, AI_MODULE - 1))
  
Event:
  'Calculate times for analog output update period 
  new_time = Read_Timer()
  time_diff = Calc_TicksToNs(new_time - old_time)
  
  'Digital Output 
  If (P2_Digout_FIFO_Empty(DIO_MODULE) > Par_32 - 1) Then
    P2_Digout_Fifo_Write(DIO_MODULE, Par_32, DATA_30, Par_33)
  EndIf
  
  'Analog Output  
  If (time_diff > Par_50) Then
    P2_DAC8_Packed(AO_MODULE, DATA_50, 4 * ao_index - 3)
    old_time = Read_Timer_Sync()
    inc(ao_index)
  EndIf
  
  If ((4 * ao_index - 3) > 9997) Then
    ao_index = 1
  EndIf
    
  'Read Analog Input
  
  SelectCase Par_41
    Case 1
      P2_Burst_CRead_Unpacked1(AI_MODULE, Par_42 / 2, Data_40, 1, 3)
    Case 3
      P2_Burst_CRead_Unpacked2(AI_MODULE, Par_42 / 2, Data_40, Data_41, 1, 3)
    Case 15
      P2_Burst_CRead_Unpacked4(AI_MODULE, Par_42 / 2, Data_40, Data_41, Data_42, Data_43, 1, 3)
    Case 255
      P2_Burst_CRead_Unpacked8(AI_MODULE, Par_42 / 2, Data_40, Data_41, Data_42, Data_43, Data_44, Data_45, Data_46, Data_47, 1, 3)
  EndSelect
Finish:
  'Turn off LEDs
  P2_Set_LED(AI_MODULE, 0)
  P2_Set_LED(AO_MODULE, 0)
  P2_Set_LED(DIO_MODULE, 0)
