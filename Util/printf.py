import numpy
numpy.set_printoptions(precision = 8, linewidth = 300)  #Makes the arrays print nicely in the output

def print_to_file(outfile,string):
    try:
       outfile.write(string) 
    except:
       print(string)

#-----------------------------------------------------------------

def double_delimiter(outfile=None):
   string = '=======================================================\n'
   if outfile is None:
      return string
   else:
      print_to_file(outfile,string+'\n')

def delimiter(outfile):
   string = '-------------------------------------------------------\n'
   if outfile is None:
      return string
   else:
      print_to_file(outfile,string+'\n')

def blank_line(outfile):
   string = '\n'
   if outfile is None:
      return string
   else:
      print_to_file(outfile,string)

def format_text_value(text,value):
   if isinstance(value,numpy.ndarray): 
      sep = '\n'; end = '\n\n' 
   else:
      sep = ' '; end = '\n'
   string = text + sep + str(value) + end
   return string

def text_value(outfile,text1,value1,text2=None,value2=None,text3=None,value3=None,text4=None,value4=None,text5=None,value5=None):
   # Construct or print up to 5 sections of text and values, text and values supplied as arguments 
   string = format_text_value(text1,value1)
   if text2 is not None and value2 is not None: string += format_text_value(text2,value2)
   if text3 is not None and value3 is not None: string += format_text_value(text3,value3)
   if text4 is not None and value4 is not None: string += format_text_value(text4,value4)
   if text5 is not None and value5 is not None: string += format_text_value(text5,value5)
   if outfile is None:
      return string
   else:
      print_to_file(outfile,string+'\n')

def delimited_text_value(outfile,text1,value1,text2=None,value2=None,text3=None,value3=None,text4=None,value4=None,text5=None,value5=None):
   # Print up to 5 sections of text and values surrounded by delimiters, text and values supplied as arguments
   string = delimiter(None)
   string += text_value(None,text1,value1,text2,value2,text3,value3,text4,value4,text5,value5)
   string += delimiter(None)
   print_to_file(outfile,string+'\n')

def double_delimited_text_value(outfile,text1,value1,text2=None,value2=None,text3=None,value3=None,text4=None,value4=None,text5=None,value5=None):
   # Print up to 5 sections of text and values surrounded by delimiters, text and values supplied as arguments
   string = double_delimiter(None)
   string += text_value(None,text1,value1,text2,value2,text3,value3,text4,value4,text5,value5)
   string += double_delimiter(None)
   print_to_file(outfile,string+'\n')

def text(outfile,text1,text2=None,text3=None,text4=None,text5=None):
   # Construct or print up to 5 lines of text, each line supplied as a separate argument
   string = text1 + '\n'
   if text2 is not None: string += text2 + '\n'
   if text3 is not None: string += text3 + '\n'
   if text4 is not None: string += text4 + '\n'
   if text5 is not None: string += text5 + '\n'
   if outfile is None:
      return string
   else:
      print_to_file(outfile,string+'\n')

def delimited_text(outfile,text1,text2=None,text3=None,text4=None,text5=None):
   # Print up to 5 sections of text and values surrounded by delimiters, text and values supplied as arguments
   string = delimiter(None)
   string += text(None,text1,text2,text3,text4,text5)
   string += delimiter(None)
   print_to_file(outfile,string+'\n')

def double_delimited_text(outfile,text1,text2=None,text3=None,text4=None,text5=None):
   # Print up to 5 sections of text and values surrounded by delimiters, text and values supplied as arguments
   string = double_delimiter(None)
   string += text(None,text1,text2,text3,text4,text5)
   string += double_delimiter(None)
   print_to_file(outfile,string+'\n')

