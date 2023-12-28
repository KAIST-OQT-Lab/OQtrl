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

Dim Data_1[4] As Long
Init:
  Processdelay = 100
  'Turn on LED of the Modules
  P2_Set_LED(DIO_MODULE, 1)
  
  Data_1[1] = 11b
  Data_1[2] = 0
  Data_1[3] = 00b
  Data_1[4] = 400
  Par_2 = 500
  'Digital Channel Setting
  P2_DigProg(DIO_MODULE, 0Fh) 'Set Digital Input / Output Ch
  P2_Dig_FIFO_Mode(DIO_MODULE, 5)'Set DIO FIFO as Relative output
  P2_Digout_FIFO_Clear(DIO_MODULE) 'Clear FIFO 
  P2_Digout_FIFO_Enable(DIO_MODULE, 1111b) 'Make Selected Output CH as FIFO
  P2_Digout_Fifo_Write(DIO_MODULE, 2, Data_1, 1)'Write FIFO Pattern
  P2_Digout_FIFO_Start(Shift_Left(1, DIO_MODULE - 1)) 'Start Digital Output FIFO '10'
    
Event:
  Par_1 = P2_Digout_FIFO_Read_Timer(DIO_MODULE)
  'Digital Output 
  'If (P2_Digout_FIFO_Empty(DIO_MODULE) > 2) Then
  'P2_Digout_FIFO_Clear(DIO_MODULE)
  'P2_Digout_FIFO_Write(DIO_MODULE, 2, Data_1, 1)
  'P2_Digout_FIFO_Start(Shift_Left(1, DIO_MODULE - 1))
  'EndIf
  
  If (P2_Digout_FIFO_Read_Timer(DIO_MODULE) > Par_2-1) Then
    P2_Digout_FIFO_Clear(DIO_MODULE)
  P2_Digout_FIFO_Write(DIO_MODULE, 4, Data_1, 1)
  P2_Digout_FIFO_Start(Shift_Left(1, DIO_MODULE - 1))
EndIf
  
  
