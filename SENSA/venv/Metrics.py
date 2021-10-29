import os
import sys

#sys.path.insert(0, "C:/program files/scitools/bin/pc-win64/python")
os.add_dll_directory('D:/Program Files/SciTools/bin/pc-win64')
#import understand as und

import subprocess
import re
import csv

import pandas as pd
from naming import UnderstandUtility
#import understand
class cls_get_metrics:

    def is_accesor(self, input_method):
        try:
            for ret in input_method.contents():
                ret = str(ret).replace(" ", "")
            if (str(input_method.simplename()).startswith(("get","Get")) and  len(self.return_parameters(input_method))==0   and len(input_method.refs('Set'))==0):
               attrebutes = self.return_attributes_type(input_method.parent())
               methodtype = str(self.return_type(input_method)).startswith('void')
               if (methodtype in attrebutes):
                   return True
               return True
            else:
               return False
        except:
            raise("Error")

    def is_mutator(self, input_method):
        try:
            if (str(input_method.simplename()).startswith(("set", "Set")) and len(self.return_parameters(input_method))==1  and str(self.return_type(input_method)).startswith( 'void') and len(input_method.refs('Set'))==1):
               attrebutes=self.return_attributes_longname(input_method.parent())
               for attr in input_method.refs('Set') :
                   if(attr.ent().longname() in attrebutes):
                       return True
            else:
               return False
        except:
            raise("Error")

    def is_accesor_or_mutator(self, input_method):
        try:
            if (self.is_accesor(input_method)==True or self.is_mutator(input_method)==True):
               return True
            else:
               return False
        except:
            raise("Error")

    def is_interface(self, input_class):
        try:
            if ("Interface" in str(input_class.kindname())):
                return True
            else:
                return False
        except:
            raise("Error")

    def is_abstract(self, input):
        try:
            if ("Abstract" in str(input.kind())):
                return True
            else:
                return False
        except:
            raise("Error")

    def is_abstract_or_interface(self, input_method):
        try:
            if (self.is_abstract(input_method)==True or self.is_interface(input_method)==True ):
                return True
            else:
                return False
        except:
            raise("Error")

    def number_of_parameters(self,input_method):
        try:
            number_of_parameters = 0
            params = input_method.parameters().split(',')
            for a in params:
                print(a)
            print(params)
            if len(params) == 1:
                print(len(params))
                print(params[0])
                if params[0] == ' ' or params[0] == '' or params[0] is None:
                    number_of_parameters += 0
                else:
                    number_of_parameters += 1
            else:
                number_of_parameters += len(params)
            return
        except:
            return False

    def return_parameters(self, input_method):
        try:
            parameters=list()
            for parameter in input_method.ents('Parameters'):
                if (parameter.kindname()=="Parameter"):
                    parameters.append(parameter)
            return parameters
        except:
            return False

    def return_attributes(self, input):
        try:
           return input.ents('Define','Variable')
        except:
            return False

    def return_attributes_simplename(self, input):
        try:
            attributes=list()
            for attribute in input.ents('Define', 'Variable'):
                attributes.append(attribute.simplename())
            return attributes
        except:
            return False

    def return_attributes_longname(self, input):
        try:
            attributes=list()
            for attribute in input.ents('Define', 'Variable'):
                attributes.append(attribute.longname())
            return attributes
        except:
            return False

    def return_attributes_type(self, input):
        try:
            attributes=list()
            for attribute in input.ents('Define', 'Variable'):
                attributes.append(attribute.type())
            return attributes
        except:
            return False

    def return_type(self, input_method):
        try:
            for ret in input_method.ib():
                ret = str(ret).replace(" ", "")
                if (ret.startswith('ReturnType')):
                    ret = ret.split(':')
                    return ret[1]
        except:
            return False

    def nopk(self,input_db):
        try:
            count=0
            for package in input_db.ents("Package"):
                if package.library() != "Standard":
                    count+=1
            return count
        except:
            return 0

    def nocs(self,input_class):
        try:
            count=0
            classes = input_class.ents('Define','class')
            if(len(classes)>0):
                count += len(classes)
                for cls in classes:
                 count += self.nocs(cls)
            else:
             count += 0
            return count
        except:
            return 0

    def nocs_package(self,input_package):
        try:
            return int(0 if input_package.metric(['CountDeclClass'])['CountDeclClass'] is None else input_package.metric(['CountDeclClass'])['CountDeclClass'])
        except:
            return 0

    def nocs_project(self,input_project):
        try:
            return int(0 if input_project.metric(['CountDeclClass'])['CountDeclClass'] is None else input_project.metric(['CountDeclClass'])['CountDeclClass'])
        except:
            return 0

    def nom(self,input_class):
        try:
            count = 0
            for mth in input_class.ents('Define','method'):
             if(not mth.refs('Override')):
                 count+=1
            return count
        except:
            return 0

    def nom_package(self,input_package,input_db):
        try:
            count = 0
            classes = input_db.ents('class')
            if (len(classes) > 0):
                for cls in classes:
                    clsname=cls.longname()
                    if(str(input_package.name()) == clsname[0:len(input_package.name())]):
                     count += self.nom(cls)
            else:
                count += 0
            return count
        except:
            return 0

    def nom_project(self,input_db):
        try:
            count = 0
            for package in input_db.ents('package'):
                count+=self.nom_package(package,input_db)
            return count
        except:
            return 0

    def noa(self, input_class):
        try:
            count = 0
            entities = input_class.ents('Define','Variable')
            return len(entities)
        except:
            return 0

    def wmc(self, input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                return int(0 if input_class.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'] is None else input_class.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'])
        except:
            return 0

    def amw(self, input_class):
            if (self.is_interface(input_class)):
                return 0
            else:
                return int(0 if input_class.metric(['AvgCyclomaticStrict'])['AvgCyclomaticStrict'] is None else input_class.metric(['AvgCyclomaticStrict'])['AvgCyclomaticStrict'])

    def wmcnamm(self,input_class):
        try:
            if(self.is_interface(input_class)):
                return 0
            else:
                sum=0
                for mth in input_class.ents('Define','method'):
                    if  not(self.is_accesor_or_mutator(mth)):
                        if(mth.metric(["CyclomaticStrict"])["CyclomaticStrict"] != None):
                            sum += mth.metric(["CyclomaticStrict"])["CyclomaticStrict"]
                return sum
        except:
            return 0

    def noam(self,input_class):
        try:
            if(self.is_interface(input_class)):
                return 0
            else:
                count = 0
                for mth in input_class.ents('Define', 'method'):
                    if (self.is_accesor_or_mutator(mth)==True):
                        count += 1
                return count
        except:
            return 0

    def NOMAMM(self, class_entity):
        """
        Number of accessor (getter) and mutator (setter) methods
        Weak implementation, need to rewrite
        :param class_entity:
        :return:
        """
        count = 0
        methods = class_entity.ents('Define', 'Java Method')
        if methods is not None:
            for method_entity in methods:
                # if str(self.return_name(method_entity)).startswith(('get', 'Get', 'GET', 'set', 'Set', 'SET')):
                #     count += 1
                if str(method_entity.simplename()).startswith(('get', 'Get', 'GET', 'set', 'Set', 'SET')):
                    count += 1  # i.e., the method is accessor or mutator
        return count

    def nomnamm(self, input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                count = 0
                for mth in input_class.ents('Define', 'method'):
                    if (self.is_accesor_or_mutator(mth) == False):
                        count += 1
                return count
        except:
            return 0

    def amwnamm(self,input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                cint = self.nomnamm(input_class)
                if cint == 0:
                    return 0
                else:
                    return (self.wmcnamm(input_class)/cint)
        except:
            return 0


    def nop(self,input_method):
        try:
             count=0
             for Parameter in input_method.ents('Parameters'):
                if(str(Parameter.kindname())=="Parameter"):
                 count+=1
             return count
        except:
            return 0

    def dit(self,input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                return int(0 if input_class.metric(['MaxInheritanceTree'])['MaxInheritanceTree'] is None else input_class.metric(['MaxInheritanceTree'])['MaxInheritanceTree'])
        except:
             return 0

    def noc(self,input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                return int(0 if input_class.metric(['CountClassDerived'])['CountClassDerived'] is None else input_class.metric(['CountClassDerived'])['CountClassDerived'])
        except:
             return 0

    #def nim(self,input_class):
     #   try:
      #      if (self.is_abstract(input_class)):
       #         return 0
        #    else:
         #       CountDeclMethod = int(0 if input_class.metric(['CountDeclMethod'])['CountDeclMethod']is None else input_class.metric(['CountDeclMethod'])['CountDeclMethod'])
          #      CountDeclMethodAll = int(0 if input_class.metric(['CountDeclMethodAll'])['CountDeclMethodAll'] is None else input_class.metric(['CountDeclMethodAll'])['CountDeclMethodAll'])
           #     return CountDeclMethodAll-CountDeclMethod
        #except:
         #   return 0

    def NIM(self, class_name):
        CountDeclMethod = int(0 if class_name.metric(['CountDeclMethod'])['CountDeclMethod'] is None else
                              class_name.metric(['CountDeclMethod'])['CountDeclMethod'])
        CountDeclMethodAll = int(0 if class_name.metric(['CountDeclMethodAll'])['CountDeclMethodAll'] is None else
                                 class_name.metric(['CountDeclMethodAll'])['CountDeclMethodAll'])
        return CountDeclMethodAll - CountDeclMethod


    def lcom5(self, input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                result=0
                NOACC=0
                NOM = self.nom(input_class)
                NOA = self.noa(input_class)
                entities = input_class.ents('Define','Variable')
                for enty in entities:
                    for ref in enty.refs('Useby'):
                        if((ref.ent().kind()!="Constructor")):
                         NOACC+=1

                if(NOM>1 and NOA>0):
                 result =(NOM-(NOACC/NOA))/(NOM-1)
                else:
                 result=0
            return result
        except:
            return 0

    def FANIN(self, db=None, class_entity=None) -> int:
        """
        Method for computing the fanin for a given class
        :param db: Understand database of target project
        :param class_entity: Target class entity for computing fanin
        :return: fain: The fanin of the class
        """
        method_list = UnderstandUtility.get_method_of_class_java(db=db, class_name=class_entity.longname())
        fanin = 0
        for method_entity in method_list:
            method_fanin = method_entity.metric(['CountInput'])['CountInput']
            if method_fanin is None:
                fanin += 0
            else:
                fanin += method_fanin
        return fanin

    def FANIN_Method(self, methodname):
        try:
            if (self.is_abstract(methodname) or self.is_interface(methodname.parent())):
                return 0
            else:
                called_classes_set = set()
                for call in methodname.refs("callby"):
                    if (call.ent().library() != "Standard"):
                        called_classes_set.add(call.ent().parent().longname())
                return len(called_classes_set)
        except:
            return 0

    def FANOUT(self, db=None, class_entity=None) -> int:
        method_list = UnderstandUtility.get_method_of_class_java(db=db, class_name=class_entity.longname())
        fanout = 0
        for method_entity in method_list:
            method_fanout = method_entity.metric(['CountOutput'])['CountOutput']
            if method_fanout is None:
                fanout += 0
            else:
                fanout += method_fanout
        return fanout

    def FANOUT_Method(self, funcname):
        try:
            if (self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return (funcname.metric(["CountOutPut"])["CountOutPut"])
        except:
            return None

    def cbo(self, input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                return int(0 if input_class.metric(['CountClassCoupled'])['CountClassCoupled'] is None else input_class.metric(['CountClassCoupled'])['CountClassCoupled'])
        except:
            return 0


    def noi_project(self, input_db):
        try:
            return len(input_db.ents('interface'))
        except:
            return 0

    def noi_package(self,input_package, input_db):
        try:
            count=0
            for interface in input_db.ents('interface'):
                interfacename = interface.longname()
                if (str(input_package.name()) == interfacename[0:len(input_package.name())]):
                 count += 1
            return count
        except:
            return 0

    def nmo(self, input_class):
        try:
             um = 0
             for mth in input_class.ents('Define', 'method'):
                if (mth.refs('Override')):
                    um += 1
             return um
        except:
            return 0

    def noii(self, input_class):
        try:
            count=0
            for b in input_class.ents('Implement'):
                if(b.kindname()=="Interface"):
                 count+=1
            return count
        except:
            return 0

    def clnamm(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                count = 0
                print('Point 1 ' + str(datetime.now()))
                for meth in input_method.refs('call'):
                    if(str(meth.ent().parent().longname())==str(input_method.parent().longname()) and not (self.is_accesor_or_mutator(meth.ent()))):
                          count+=1
                return count
                print('Point 1 ' + str(datetime.now()))
        except:
            return 0

    def fdp(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                listmethod = set()
                for meth in input_method.refs('Use', 'Variable'):
                    if (meth.ent().parent().longname() != input_method.parent().longname()):
                            listmethod.add(meth.ent().parent().longname())
                return len(listmethod)
        except:
            return 0


    def rfc(self, input_class):
        try:
            if (self.is_interface(input_class)):
                return 0
            else:
                count = 0
                listmethod=set()
                for meth in input_class.refs('call', 'method'):
                    if (meth.ent().parent().longname() != input_class.longname() and meth.ent().kindname()!= "Public Constructor"):
                      listmethod.add(meth.ent().longname())
                return len(listmethod)
        except:
            return 0

    def laa(self,input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                print('Point 1 ' + str(datetime.now()))
                result = 0
                listtotal = set()
                listsameclass = set()
                print('Point 2 ' + str(datetime.now()))
                for vr in input_method.refs('Use', 'Variable'):
                    listtotal.add(vr.ent().longname())
                    if (vr.ent().parent().longname() == input_method.parent().longname()):
                            listsameclass.append(vr.ent().longname())
                print('Point 3 ' + str(datetime.now()))
                for meth in input_method.refs('call', 'method'):
                        if meth.ent().kindname() != "Public Constructor":
                            if (self.is_accesor_or_mutator(meth.ent())):
                                listtotal.add(meth.ent().longname())
                            if (self.is_accesor_or_mutator(meth.ent()) and meth.ent().parent().longname() == input_method.parent().longname()):
                                listsameclass.add(meth.ent().longname())
                print('Point 4 ' + str(datetime.now()))
                if (len(listtotal) != 0):
                    result = len(listsameclass) / len(listtotal)
                return result
        except:
            return 0

    def cfnamm_method(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                listmethod = list()
                for meth in method_name.refs('call', 'method'):
                    if (meth.ent().parent().longname() != input_method.parent().longname() and meth.ent().kindname() != "Public Constructor" and not (self.is_accesor_or_mutator(meth.ent()))):
                        listmethod.append(meth)
                return len(listmethod)
        except:
            return 0

    def CFNAMM_Class(self, class_name):
        list1 = list()
        for meth in class_name.refs('Call', 'Java Method'):
            if meth.ent().parent() is None:
                continue
            if meth.ent().parent().longname() != class_name.longname() \
                    and meth.ent().kindname() != "Public Constructor" \
                    and not (self.is_accesor_or_mutator(meth.ent())):
                list1.append(meth)
        return len(set(list1))

    def noav(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                print('Point 1 ' + str(datetime.now()))
                count = 0
                for mth in input_method.refs('Use', 'Variable'):
                        count += 1
                print('Point 2 ' + str(datetime.now()))
                for meth in input_method.refs('call', 'method'):
                        if meth.ent().kindname() != "Public Constructor":
                            if (self.is_accesor_or_mutator(meth.ent())):
                                for mth in meth.ent().refs('Use', 'Variable'):
                                    count += 1
                print('Point 3 ' + str(datetime.now()))
                return count
        except:
            return 0

    def get_family_list(self, classname):
        family = list()
        try:
            if (len(classname.refs("Base")) == 0):
                family.append(classname)
                return family
        except:
            return family
        family_list = list()
        family_list.append(classname)
        for cla in family_list:
            for f in cla.refs("Base,Derive"):
                if not (f.ent() in family_list) and not (len(f.ent().refs("Base")) == 0):
                    family_list.append(f.ent())
        return family_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def get_childs(self, classname):
        child_list = list()
        try:
            for f in classname.refs("Derive"):
                child_list.append(f.ent())
            return child_list
        except:
            return child_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def get_fathers_and_grandfathers(self, classname):
        fathers_list = list()
        try:
            fathers_list.append(classname)
            while (True):
                for f in classname.refs("Base"):
                    parent = f.ent()
                    print("f:", f)
                print("parent        :", parent.name())
                if (parent.name() == "Object" or parent.name() == "page"):
                    break
                classname = parent
                fathers_list.append(classname)
            return fathers_list
        except:
            return fathers_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def NOPA(self, classname):
        try:
            if (classname.language() == "Java"):
                return len(classname.ents("define", "Java Variable Public Member"))
            else:
                return classname.metric(["CountDeclInstanceVariablePublic"])["CountDeclInstanceVariablePublic"]
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def MAXNESTING(self, funcname):
        try:
            if (self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return funcname.metric(["MaxNesting"])["MaxNesting"]
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

   # def FANOUT(self, funcname):
    #    try:
     #       if (self.is_abstract(funcname) or self.is_interface(funcname.parent())):
      #          return None
       #     else:
        #        return (funcname.metric(["CountOutPut"])["CountOutPut"])
        #except:
         #   return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def CYCLO(self, funcname):
        try:
            if (self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return funcname.metric(["Cyclomatic"])["Cyclomatic"]
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # def is_abstract(self,funcname):
        #     if (str(funcname).startswith(("get", "set", "Set", "Get"))   or    funcname.metric(["CountLine"])["CountLine"]<=6):
        #         return True
        #     else:
        #         return False

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def NOLV(self, funcname):
        try:
            if (self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                # bug
                varlist = funcname.ents("", "Variable , Parameter")
                return len(varlist)
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def NOAM(self, class_name):
        try:
            if (self.is_interface(class_name)):
                #print('sss')
                return 0
            else:
                count = 0
                for mth in class_name.ents('Define', 'method'):
                    if (str(mth.name()).startswith(("get", "set", "Set", "Get"))):
                        # print(mth.longname())
                        count += 1
                        # print(mth.longname())
                return count
        except:
            return None

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def NOMNAMM(self, class_name):
        try:
            #print('_'*75)
            #print(class_name)
            mth_ = class_name.ents('Define', 'Java Method ~Unknown ~Unresolved ~Jar ~Library')
            #print(mth_)
            return ((len(mth_)) - self.NOAM(class_name))
        except Exception as e:  # work on python 3.x
            #print('Failed to upload to ftp: ' + str(e))
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def LOCNAMM(self, class_name):
        try:
            LOC = class_name.metric(["CountLine"])["CountLine"]
            if (LOC == None):
                return None
            else:
                LOCAMM = 0
                for mth in class_name.ents('Define', 'method'):
                    if (str(mth).startswith(("get", "set", "Set", "Get"))):
                        if (mth.metric(["CountLine"])["CountLine"] != None):
                            LOCAMM += mth.metric(["CountLine"])["CountLine"]
                return (LOC - LOCAMM)
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def LOC(self, funcname):
        try:
            return funcname.metric(["CountLine"])["CountLine"]
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def WOC(self, class_name):
        # try:
        #     if(self.is_interface(class_name) or self.is_abstract(class_name)):
        #         return None
        #     else:
        count_functionl = 0
        metlist = class_name.ents("Define", "Public Method")
        for m in metlist:
            if not (self.is_abstract(m)):
                count_functionl += 1
        # print(class_name.metric(["CountDeclInstanceVariablePublic"])["CountDeclInstanceVariablePublic"])
        # print(class_name.longname() ,"  ",class_name.ents("","Java Variable Public Member"))
        # baraye bazi az class hayy ke be an dastersi nadarad meghdar None bar migardand va mohasebat error mishavad
        if (class_name.metric(["CountDeclInstanceVariablePublic"])["CountDeclInstanceVariablePublic"] == None or
                class_name.metric(["CountDeclMethodPublic"])["CountDeclMethodPublic"] == None):
            return None
        else:
            total = class_name.metric(["CountDeclInstanceVariablePublic"])["CountDeclInstanceVariablePublic"] + \
                    class_name.metric(["CountDeclMethodPublic"])["CountDeclMethodPublic"]
            if total == 0:
                return 0
            else:
                return (count_functionl / total)
        # except:
        #     return None

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def WMCNAMM(self, class_name):
        try:
            if (self.is_interface(class_name)):
                return None
            else:
                sum = 0
                for mth in class_name.ents('Define', 'method'):
                    if not (self.is_accesor_or_mutator(mth)):
                        if (mth.metric(["Cyclomatic"])["Cyclomatic"] != None):
                            sum += mth.metric(["Cyclomatic"])["Cyclomatic"]
                return sum
        except:
            return None

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def TCC(self, class_name):
        try:
            if (self.is_abstract(class_name) or self.is_interface(class_name)):
                return None
            else:
                # cal NP
                NDC = 0
                methodlist = class_name.ents('Define', 'Public Method')
                method_list_visible = list()
                for mvvisible in methodlist:
                    if self.is_visible(mvvisible):
                        method_list_visible.append(mvvisible)
                for row in range(0, len(method_list_visible)):
                    for col in range(0, len(method_list_visible)):
                        if (row > col):
                            if (self.connectivity(method_list_visible[row], method_list_visible[col])):
                                NDC += 1
                N = len(method_list_visible)
                NP = N * (N - 1) / 2
                if (NP != 0):
                    return NDC / NP
                else:
                    return 0
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def connectivity(self, row, col):
        try:
            if (self.connectivity_directly(row, col) or self.connectivity_indirectly(row, col)):
                return True
            else:
                return False
        except:
            return False
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def connectivity_indirectly(self, row, col):
        listrow = set()
        listcol = set()
        try:
            for callrow in row.refs("call"):
                if (str(callrow.ent().name()).startswith(("get", "Get"))):
                    listrow.add(callrow.ent().longname())
            for callcol in col.refs("call"):
                if (str(callcol.ent().name()).startswith(("get", "Get"))):
                    listcol.add(callcol.ent().longname())
            intersect = [value for value in listrow if value in listcol]
            if (len(intersect) > 0):
                return True
            else:
                return False
        except:
            return False
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def connectivity_directly(self, row, col):
        listrow = set()
        listcol = set()
        try:
            for callrow in row.refs("use"):
                listrow.add(callrow.ent().longname())
            for callcol in col.refs("use"):
                listcol.add(callcol.ent().longname())
            intersect = [value for value in listrow if value in listcol]
            if (len(intersect) > 0):
                return True
            else:
                return False
        except:
            return False
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def is_visible(self, funcname):
        try:
            flag = False
            # """all parameter not  only use or declare """
            par = funcname.ents("", "Parameter")
            for p in par:
                if (str(p.type()) == "EventArgs"):
                    flag = True
                    break
            if not (str(funcname.kind()) == "Public Constructor") or not (flag) or not (
                    str(funcname.kind()) == "Private Method"):
                return True
            else:
                return False
        except:
            return False
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def CDISP(self, method_name):
        try:
            if (self.is_abstract(method_name) or self.is_interface(method_name.parent())):
                return None
            else:
                cint = self.CINT(method_name)

                if cint == 0:
                    return 0
                elif (cint == None):
                    return None
                else:
                    return self.FANOUT_OUR(method_name) / cint
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def FANOUT_OUR(self, methodname):
        try:
            if (self.is_abstract(methodname) or self.is_interface(methodname.parent())):
                return None
            else:
                called_classes_set = set()
                for call in methodname.refs("call"):
                    if (call.ent().library() != "Standard"):
                        called_classes_set.add(call.ent().parent())
                return len(called_classes_set)
        except:
            return None

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def ATLD(self, db, method_name):
        try:
            if (not (self.is_abstract(method_name)) or not (self.is_interface(method_name.parent()))):
                # cal directly access
                count = 0
                system_att_ = db.ents('field')
                access_att_ = self.give_access_use(method_name)
                for att_ in access_att_:
                    if att_ in system_att_:
                        if (str(att_.kind()) in ["Unknown Variable", "Unknown Class"]):
                            continue
                        if (att_.library() != "Standard"):
                            count += 1
                # cal indirectly access
                calls = self.give_ALL_sys_and_lib_method_that_th_measyred_method_calls(method_name)
                for call in calls:
                    if (str(call).startswith(("get", "Get"))):
                        usevariable = call.refs("use")
                        if (len(usevariable) > 0):
                            flag = True
                            for us in usevariable:
                                if (us.ent().library() == "Standard"):
                                    flag = False
                            if (flag):
                                count += 1
                return count
            else:
                return None
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_ALL_sys_and_lib_method_that_th_measyred_method_calls(self, funcname):

        methodlist = set()
        try:
            for refmethod in funcname.refs("call"):
                methodlist.add(refmethod.ent())
            return methodlist
        except:
            return methodlist
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def get_Namespace(self, entity):
        while (str(entity.parent().kind()) != "Unresolved Namespace" or str(entity.parent().kind()) != "Namespace"):
            entity = entity.parent()
            if (str(entity.parent().kind()) == "Unresolved Namespace" or str(entity.parent().kind()) == "Namespace"):
                break
        return entity.parent()
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def ATFD_method(self, db, method_name):
        try:
            if not (str(method_name).startswith("get", "set", "Get", "Set")):
                # cal directly access
                count = 0
                system_att_ = db.ents('Variable')
                access_att_ = self.give_access_use(method_name)
                for att_ in access_att_:
                    if not (att_ in system_att_):
                        count += 1
                # cal indirectly access
                calls = give_Methods_that_the_measured_method_calls(self, method_name)
                for call in calls:
                    if (str(call).startswith(("get", "set", "Set", "Get"))):
                        get_att_ = self.give_access_use(call)
                        if not (get_att_ in system_att_):
                            count += 1
                return count
            else:
                return 0
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # checked ok successfully ( ckeck field of librabry ex:math.PI isincluded or not ? at :if(var_.parent() not in family_list and var_.library()!="Standard"   ):

    def ATFD_CLASS(self, class_namee):
        try:
            count = 0
            family_list = self.get_family_list(class_namee)
            varibleset = set()
            methodlist = class_namee.ents('Define', 'Public Method')
            for methodname in methodlist:
                # directly
                if (not (self.is_abstract(methodname)) and methodname.name() != class_namee.name()):
                    method_accessvariable = self.give_access_use(methodname)
                    for var_ in method_accessvariable:
                        if ("Field" in str(var_.kind())):
                            if (var_.parent() not in family_list and var_.library() != "Standard"):
                                varibleset.add(var_)
                                count += 1
                    # indirectly
                    method_called_list = self.give_Methods_that_the_measured_method_calls(methodname)
                    # print(method_called_list)
                    for m in method_called_list:
                        # print(m.parent())
                        if (m.parent() not in family_list and str(m).startswith(("get", "Get"))):
                            varibleset.add(m)
                            count += 1
            return len(varibleset)
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def max_depth(self, method_name):
      try:
        # at least two
        if self.len(give_Methods_that_the_measured_method_calls(method_name)) == 0:
            return 0
        # if self.give_Methods_that_the_measured_method_calls(method_name).count() ==1:
        #    return 1
        return 1 + max([self.Mamcl(node) for node in self.give_Methods_that_the_measured_method_calls(method_name)])
      except:
          return  None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def total_nods(self, method_name):
     try:
        if self.give_Methods_that_the_measured_method_calls(method_name).count() == 0:
            return 0
        if self.give_Methods_that_the_measured_method_calls(method_name).count() == 1:

            return 1
        return 1 + sum([self.Mamcl(node) for node in self.give_Methods_that_the_measured_method_calls(method_name)])
     except:
         return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def NMCS(self, method_name):
        try:
            count = 0
            for mth in self.give_Methods_that_the_measured_method_calls(method_name):
                if str(mth).startswith("get", "set", "Get", "Set"):
                    count += 1
            return count
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def MaMCL(self, method_name):
        try:
            max_dep = self.max_depth(method_name)
            if max_dep >= 2:
                return max_dep
            else:
                return 0
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def Memcl(self, method_name):
        try:
            total = self.total_nods(method_name)
            nmcs = self.NMCS(method_name)
            if nmcs == 0:
                return 0
            else:
                return round(total / nmcs)
        except:
            return None

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # def CC(self,funcname):
        #     refset = set()
        #     reflist = list()
        #     for callint in funcname.refs("callby"):
        #         refset.add(callint.ent().parent())
        #         #reflist.append(callint.ent().parent())
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def CM(self, funcname):
        try:
            if (self.is_private(funcname)):
                return None
            else:
                refset = set()
                for callint in funcname.refs("callby"):
                    refset.add(callint.ent())
                return len(refset)
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def is_private(slef, funcname):
        try:
            if (str(funcname.kind()) == "Private Method"):
                return True
            else:
                return False
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def CINT(self, method_name):
        try:
            if (self.is_abstract(method_name) or self.is_interface(method_name.parent())):
                return None
            else:
                count = 0
                family_list = self.get_family_list(method_name.parent())
                for mth in method_name.refs("call"):
                    if (mth.ent().parent() in family_list):
                        count += 1
                return count
        except:
            return None

    def CountLineBlank(self, funcname):
            try:
                    return funcname.metric(["CountLineBlank"])["CountLineBlank"]
            except:
                return None

    def NL(self, funcname):
            try:
                return funcname.metric(["CountLine"])["CountLine"]
            except:
                return None

    def CountLineCodeExe(self, funcname):
            try:
                return funcname.metric(["CountLineCodeExe"])["CountLineCodeExe"]
            except:
                return None


    def NPATH(self, funcname):
            try:
                return funcname.metric(["CountPath"])["CountPath"]
            except:
                return None

    def CountStmt(self, funcname):
            try:
                return funcname.metric(["CountStmt"])["CountStmt"]
            except:
                return None

    def CountStmtDecl(self, funcname):
            try:
                return funcname.metric(["CountStmtDecl"])["CountStmtDecl"]
            except:
                return None


    def CountStmtExe(self, funcname):
            try:
                return funcname.metric(["CountStmtExe"])["CountStmtExe"]
            except:
                return None


    def CyclomaticStrict(self, funcname):
            try:
                return funcname.metric(["CyclomaticStrict"])["CyclomaticStrict"]
            except:
                return None


    def CyclomaticModified(self, funcname):
            try:
                return funcname.metric(["CyclomaticModified"])["CyclomaticModified"]
            except:
                return None


    def Essential(self, funcname):
            try:
                return funcname.metric(["Essential"])["Essential"]
            except:
                return None


    def Knots(self, funcname):
            try:
                return funcname.metric(["Knots"])["Knots"]
            except:
                return None


    def MaxEssentialKnots(self, funcname):
            try:
                return funcname.metric(["MaxEssentialKnots"])["MaxEssentialKnots"]
            except:
                return None


    def RatioCommentToCode(self, funcname):
            try:
                return funcname.metric(["RatioCommentToCode"])["RatioCommentToCode"]
            except:
                return None


    def CountPathLog(self, funcname):
            try:
                return funcname.metric(["CountPathLog"])["CountPathLog"]
            except:
                return None

    def CountLineCodeDecl(self, funcname):
            try:
                return funcname.metric(["CountLineCodeDecl"])["CountLineCodeDecl"]
            except:
                return None


        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_access_use(self, funcname):
        # create a list and return it:Includes all the variables(fields) that a method uses
        access_field_list = set()
        try:
            for fi in funcname.refs("use"):
                access_field_list.add(fi.ent())
            return access_field_list
        except:
            return access_field_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_access_use_for_class(self, classname):
        # create a list and return it:Includes all the variables(fields) that a method uses
        access_field_list = list()
        try:
            for fi in classname.refs("use"):
                access_field_list.append(fi.ent())
            return access_field_list
        except:
            return access_field_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_Methods_that_the_measured_method_calls(self, funcname):
        call_methods_list = set()
        try:
            for fi in funcname.refs("call"):
                if (fi.ent().library() != "Standard"):
                    call_methods_list.add(fi.ent())
            return call_methods_list
        except:
            return call_methods_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_cc(self, db, funcname):
        try:
            if (self.is_private(funcname)):
                return None
            else:
                refset = set()
                for callint in funcname.refs("callby"):
                    refset.add(callint.ent().parent())
                return len((refset))
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def give_Methods_that_the_measured_class_calls(self, classname):
        # create a list and return it:Includes all Methods entity(also cunstructor method ) that the measured method calls
        call_methods_list = list()
        try:
            for fi in classname.refs("call"):
                # if namespace == method namespace
                if (fi.ent().parent().parent() == classname.parent()):
                    call_methods_list.append(fi.ent())
            return call_methods_list
        except:
            return call_methods_list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def returt_result(self, db):
        self.get_metrics(db)
        return [self.class_metrics, self.method_metrics]
        # return a list consist of classes and methods and thier metrics value

class PreProcess:

    def create_understand_database_from_project(cls, root_path=None):
        cmd = 'und create -db {0}{1}.udb -languages java add {2} analyze -all'
        print(cmd)
        projects = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        for project_ in projects:
            print(project_)
            command_ = cmd.format(root_path, project_, root_path + project_)
            print('cmd /c "{0}"'.format(command_))
            os.system('cmd /c "{0}"'.format(command_))


#obj = PreProcess();
#obj.create_understand_database_from_project('C:/Users/moham/Desktop/javaexample/')
