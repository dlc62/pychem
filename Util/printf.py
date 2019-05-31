
# Make numpy print array properly 
import numpy
import sys 
numpy.set_printoptions(precision = 8,
                       linewidth = sys.maxsize,
                       threshold = sys.maxsize,
                       )  

# TODO rewrite all of these using variadic argument function

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

def text_value(outfile, *pairs):
    string = ""
    # Loop over and print each pair of text and values
    for text, value in zip(pairs[0::2], pairs[1::2]):
        if text is not None and value is not None:
            string += format_text_value(text, value)
    if outfile is None:
        return string
    else:
        print_to_file(outfile, string+'\n')

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

def text(outfile, *texts):
    """ Construct or print lines of text, each provided as a seperate argument"""
    string = ""
    for text in texts:
        if text is not None:
            string += text + '\n'
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

