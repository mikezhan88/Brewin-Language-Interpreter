import copy
from enum import Enum
from env_v2 import EnvironmentManager, SymbolResult
from func_v2 import FunctionManager
from intbase import InterpreterBase, ErrorType
from tokenize import Tokenizer

# Enumerated type for our different language data types
class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  VOID = 4
  FUNC = 5
  OBJECT = 6

# Represents a value, which has a type and its value
class Value:
  def __init__(self, type, value = None):
    self.t = type
    self.v = value

  def value(self):
    return self.v

  def set(self, other):
    self.t = other.t
    self.v = other.v

  def type(self):
    return self.t

# Main interpreter class
class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()  # setup all valid binary operations and the types they work on
    self._setup_default_values()  # setup the default values for each type (e.g., bool->False)
    self.trace_output = trace_output

  # run a program, provided in an array of strings, one string per line of source code
  def run(self, program):
    #print(program)
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self.ip = self.func_manager.get_function_info(InterpreterBase.MAIN_FUNC).start_ip
    self.return_stack = []
    self.terminate = False
    self.env_manager = EnvironmentManager()   # used to track variables/scope

    # main interpreter run loop
    while not self.terminate:
      self._process_line()

  def _process_line(self):
    if self.trace_output:
      print(f"{self.ip:04}: {self.program[self.ip].rstrip()}")
    tokens = self.tokenized_program[self.ip]
    if not tokens:
      self._blank_line()
      return

    args = tokens[1:]

    match tokens[0]:
      case InterpreterBase.ASSIGN_DEF:
        self._assign(args)
      case InterpreterBase.FUNCCALL_DEF:
        self._funccall(args)
      case InterpreterBase.ENDFUNC_DEF:
        self._endfunc()
      case InterpreterBase.IF_DEF:
        self._if(args)
      case InterpreterBase.ELSE_DEF:
        self._else()
      case InterpreterBase.ENDIF_DEF:
        self._endif()
      case InterpreterBase.RETURN_DEF:
        self._return(args)
      case InterpreterBase.WHILE_DEF:
        self._while(args)
      case InterpreterBase.ENDWHILE_DEF:
        self._endwhile(args)
      case InterpreterBase.VAR_DEF: # v2 statements
        self._define_var(args)
      case InterpreterBase.LAMBDA_DEF:
        self._lambda()
      case InterpreterBase.ENDLAMBDA_DEF:
        self._endlambda()
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()

  def _lambda(self):
    whole_env = self.env_manager.environment
    enclosing_envs = copy.deepcopy(whole_env[-1])
    lambda_name = self.func_manager.create_lambda_name(self.ip)
    value_tuple = (lambda_name, enclosing_envs)
    value_type = Value(Type.FUNC, value_tuple)
    self._set_result(value_type)
    lambda_indent = self.indents[self.ip]
    for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDLAMBDA_DEF and lambda_indent == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endlambda", self.ip)

  def _endlambda(self):
    self._endfunc()

  def _assign(self, tokens):
   #print('did assign')
   if len(tokens) < 2:
     super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement")
   vname = tokens[0]
   val = self.env_manager.get(vname)
   if val == None: #could be an object member variable so check for that
    if "." in vname:
      member = vname.split(".")
      obj = member[0]
      #print(f'obj name: {obj}')
      member_var = member[1]
      obj_value = self._get_value(obj)
      #print(obj_value)
      if obj_value.type() != Type.OBJECT:
        super().error(ErrorType.TYPE_ERROR, "can't use dot operator on non-object", self.ip)
      obj_value_value = obj_value.value()
      member_value = None
      if self.func_manager.is_function(tokens[1]):
        member_value = Value(Type.FUNC, tokens[1])
      else:
        member_value = self._eval_expression(tokens[1:])
      obj_value_value[member_var] = member_value
    else: #not obj member var so throw error
      super().error(ErrorType.NAME_ERROR, "Unknown Variable", self.ip)
   else:
    if val.type() == Type.FUNC: #check if we are trying to assign to a func var 
      if self.func_manager.is_function(tokens[1]): #check if func we are trying to pass in is a valid function
        self._set_value(vname, Value(Type.FUNC, tokens[1])) #if it is, pass in func name to existing func var
      else: #check if trying to assign from resultf which could be a func or lambda
        if tokens[1] == 'resultf':
          resultf_value = self._get_value(tokens[1])
          if isinstance(resultf_value.value(), tuple): #lambda check
            lambda_name = resultf_value.value()[0]
            if self.func_manager.is_function(lambda_name):
              self._set_value(vname, resultf_value)
            else: 
              super().error(ErrorType.NAME_ERROR, "need to assign from a func", self.ip)
          else:
            func_name = resultf_value.value()
            if self.func_manager.is_function(func_name):
              self._set_value(vname, Value(Type.FUNC, func_name))
            else:
              super().error(ErrorType.NAME_ERROR, "need to assign from a func", self.ip)
        else:
          super().error(ErrorType.NAME_ERROR, "need to assign from a func", self.ip)
    elif val.type() == Type.OBJECT: #could be trying to assign an object to an object var
      possible_obj = self._get_value(tokens[1])
      if isinstance(possible_obj.value(), dict):
        self._set_value(vname, possible_obj)
      else:
        super().error(ErrorType.TYPE_ERROR, "need to assign from a obj", self.ip)
    else: #if not then just process as usual
      value_type = self._eval_expression(tokens[1:])
      existing_value_type = self._get_value(tokens[0])
      if existing_value_type.type() != value_type.type():
        super().error(ErrorType.TYPE_ERROR,
                      f"Trying to assign a variable of {existing_value_type.type()} to a value of {value_type.type()}",
                      self.ip)
      self._set_value(tokens[0], value_type)
   self._advance_to_next_statement()

  def _funccall(self, args):
    print(f'funcall args: {args}')
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip)
    if args[0] == InterpreterBase.PRINT_DEF:
      self._print(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.INPUT_DEF:
      self._input(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.STRTOINT_DEF:
      self._strtoint(args[1:])
      self._advance_to_next_statement()
    else: #user defined func or func var or object method
      possible_func_var = self.env_manager.get(args[0])
      if "." in args[0]: #possible object method
        possible_obj = args[0].split(".")
        obj_name = possible_obj[0]
        obj_method = possible_obj[1]
        if self.env_manager.get(obj_name) != None: #check if obj var defined
          obj = self._get_value(obj_name)
          obj_type = obj.type()
          obj_value = obj.value()
          if obj_type == Type.OBJECT : #check type
            if obj_method in obj_value:
              if obj_value[obj_method].type() == Type.FUNC: #valid object method
                self.return_stack.append(self.ip+1)
                func_name = obj_value[obj_method].value()
                self._create_new_environment(func_name, args[1:])
                self.ip = self._find_first_instruction(func_name)

                #create this param for top block
                self.env_manager.create_new_symbol('this', True)
                self.env_manager.set('this', obj)
              else:
                super().error(ErrorType.TYPE_ERROR,"object method not of type func", self.ip)
            else: #unknown object member
              super().error(ErrorType.NAME_ERROR,"unknown object method", self.ip)
          else: #not object method
            super().error(ErrorType.NAME_ERROR,"cant use dot operator on non object method in funcall", self.ip)

      elif possible_func_var != None: #possible func var or lambda function   
        if possible_func_var.type() == Type.FUNC: #check if var is of type func
          if isinstance(possible_func_var.value(), tuple): #check if value is tuple meanings its a lambda
            self.return_stack.append(self.ip + 1)
            lambda_name = possible_func_var.value()[0]
            print(f'lambda name in funcall: {lambda_name}')
            closure = possible_func_var.value()[1]
            print(f'closure in funcall: {closure}')
            self._create_new_environment(lambda_name, args[1:], closure)
            self.ip = self._find_first_instruction(lambda_name)
          else: #valid func var
            if possible_func_var.value() == None: #if just a default func var, then do nothing and return
              return
            else: #otherwise is not a default func so go to first line of execution 
                self.return_stack.append(self.ip+1)
                func_name = possible_func_var.value()
                self._create_new_environment(func_name, args[1:])  # Create new environment, copy args into new env
                self.ip = self._find_first_instruction(func_name)
        else: 
          super().error(ErrorType.TYPE_ERROR,"cannot funcall on non function", self.ip)
      else: #user defined func
        #print("WANNA BE HERE IN FUNCALL")
        self.return_stack.append(self.ip+1)
        self._create_new_environment(args[0], args[1:])  # Create new environment, copy args into new env
        self.ip = self._find_first_instruction(args[0])

  # create a new environment for a function call
  def _create_new_environment(self, funcname, args, closure = {}):
    formal_params = self.func_manager.get_function_info(funcname)
    print(f'funcname in create new environment: {funcname}')
    print(f'formal params in create new env: {formal_params.params}')
    if formal_params is None:
        super().error(ErrorType.NAME_ERROR, f"Unknown function name {funcname}", self.ip)

    if len(formal_params.params) != len(args):
      super().error(ErrorType.NAME_ERROR,f"Mismatched parameter count in call to {funcname}", self.ip)

    tmp_mappings = {}
    for formal, actual in zip(formal_params.params,args):
      formal_name = formal[0]
      formal_typename = formal[1]
      arg = None
      if self.func_manager.is_function(actual): #actual passed in param is a func so handle appropriately
        arg = Value(Type.FUNC, actual)
      else:
        arg = self._get_value(actual)
      if arg.type() != self.compatible_types[formal_typename]:
        super().error(ErrorType.TYPE_ERROR,f"Mismatched parameter type for {formal_name} in call to {funcname}", self.ip)
      obj_type = None
      if "." in actual:
        possible_obj = actual.split(".")
        obj_name = possible_obj[0]
        obj_value = self._get_value(obj_name)
        obj_type = obj_value.type()
      if formal_typename in self.reference_types:
        tmp_mappings[formal_name] = arg
      elif obj_type == Type.OBJECT:
        tmp_mappings[formal_name] = arg
      else:
        tmp_mappings[formal_name] = copy.copy(arg)

    # create a new environment for the target function
    # and add our parameters to the env
    self.env_manager.push()
    if InterpreterBase.LAMBDA_DEF in funcname:
      for block in closure:
        self.env_manager.import_mappings(block)
    self.env_manager.import_mappings(tmp_mappings)

  def _endfunc(self, return_val = None):
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.env_manager.pop()  # get rid of environment for the function
      if return_val:
        self._set_result(return_val)
      else:
        # return default value for type if no return value is specified. Last param of True enables
        # creation of result variable even if none exists, or is of a different type
        return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
        if return_type != InterpreterBase.VOID_DEF:
          self._set_result(self.type_to_default[return_type])
      self.ip = self.return_stack.pop()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip)
    #print(f' if statement value type value: {value_type.value()}')
    if value_type.value():
      self._advance_to_next_statement()
      self.env_manager.block_nest()  # we're in a nested block, so create new env for it
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
        if tokens[0] == InterpreterBase.ELSE_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          self.env_manager.block_nest()  # we're in a nested else block, so create new env for it
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _endif(self):
    self._advance_to_next_statement()
    self.env_manager.block_unnest()

  # we would only run this if we ran the successful if block, and fell into the else at the end of the block
  # so we need to delete the old top environment
  def _else(self):
    self.env_manager.block_unnest()   # Get rid of env for block above
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _return(self,args):
    # do we want to support returns without values?
    return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
    #if return_type == InterpreterBase.OBJECT_DEF: 
      #print("prints if returning from function with return type object")
      #if 'end' in self._get_value(args[0]).value():
        #print("going in here")
        #print(self._get_value(args[0]).value()['end'].value())
      #print(f'object value: {self._get_value(args[0]).value()}')
      #obj_members = self._get_value(args[0]).value()
      #if 'next' in obj_members:
        #if 'end' in obj_members:
          #end_want = obj_members['end']
          #print(f'wahts in obj.end: {end_want.value()}')
        #want = obj_members['next']
        #print(f'whats in obj.next: {want.value()}')
        #want_next = want.value()['next'].value()
        #print(f'whats in obj.next.next: {want_next}')
        #print(f'obj attributes: {vars(self._get_value(args[0]))}')

    default_value_type = self.type_to_default[return_type]
    if default_value_type.type() == Type.VOID:
      if args:
        super().error(ErrorType.TYPE_ERROR,"Returning value from void function", self.ip)
      self._endfunc()  # no return
      return
    if not args:
      self._endfunc()  # return default value
      return

    #otherwise evaluate the expression and return its value
    if return_type == InterpreterBase.FUNC_DEF: #check if trying to return a function
      if self.func_manager.is_function(args[0]):
        self._endfunc(Value(Type.FUNC, args[0]))
        return
    #print(f'return args: {args}')
    value_type = self._eval_expression(args)
    if value_type.type() != default_value_type.type():
      super().error(ErrorType.TYPE_ERROR,"Non-matching return type", self.ip)
    self._endfunc(value_type)

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip)
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._advance_to_next_statement()
    # And create a new scope
    self.env_manager.block_nest()

  def _exit_while(self):
    while_indent = self.indents[self.ip]
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDWHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip)

  def _endwhile(self, args):
    # first delete the scope
    self.env_manager.block_unnest()
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip)


  def _define_var(self, args):
    if len(args) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid var definition syntax", self.ip)
    for var_name in args[1:]:
      if self.env_manager.create_new_symbol(var_name) != SymbolResult.OK:
        super().error(ErrorType.NAME_ERROR,f"Redefinition of variable {args[1]}", self.ip)
      # is the type a valid type?
      if args[0] not in self.type_to_default:
        super().error(ErrorType.TYPE_ERROR,f"Invalid type {args[0]}", self.ip)
      # Create the variable with a copy of the default value for the type
      self.env_manager.set(var_name, copy.deepcopy(self.type_to_default[args[0]]))

    self._advance_to_next_statement()

  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip)
    out = []
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))   # return always passed back in result

  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip)
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip)
    self._set_result(Value(Type.INT, int(value_type.value())))   # return always passed back in result

  def _advance_to_next_statement(self):
    # for now just increment IP, but later deal with loops, returns, end of functions, etc.
    self.ip += 1

  # Set up type-related data structures
  def _setup_default_values(self):
    # set up what value to return as the default value for each type
    self.type_to_default = {}
    self.type_to_default[InterpreterBase.INT_DEF] = Value(Type.INT,0)
    self.type_to_default[InterpreterBase.STRING_DEF] = Value(Type.STRING,'')
    self.type_to_default[InterpreterBase.BOOL_DEF] = Value(Type.BOOL,False)
    self.type_to_default[InterpreterBase.VOID_DEF] = Value(Type.VOID,None)
    self.type_to_default[InterpreterBase.FUNC_DEF] = Value(Type.FUNC, None) 
    self.type_to_default[InterpreterBase.OBJECT_DEF] = Value(Type.OBJECT, {})

    # set up what types are compatible with what other types
    self.compatible_types = {}
    self.compatible_types[InterpreterBase.INT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.STRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.BOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.REFINT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.REFSTRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.REFBOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.FUNC_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.OBJECT_DEF] = Type.OBJECT
    self.reference_types = {InterpreterBase.REFINT_DEF, Interpreter.REFSTRING_DEF,
                            Interpreter.REFBOOL_DEF}

    # set up names of result variables: resulti, results, resultb
    self.type_to_result = {}
    self.type_to_result[Type.INT] = 'i'
    self.type_to_result[Type.STRING] = 's'
    self.type_to_result[Type.BOOL] = 'b'
    self.type_to_result[Type.FUNC] = 'f'
    self.type_to_result[Type.OBJECT] = 'o'

  # run a program, provided in an array of strings, one string per line of source code
  def _setup_operations(self):
    self.binary_op_list = ['+','-','*','/','%','==','!=', '<', '<=', '>', '>=', '&', '|']
    self.binary_ops = {}
    self.binary_ops[Type.INT] = {
     '+': lambda a,b: Value(Type.INT, a.value()+b.value()),
     '-': lambda a,b: Value(Type.INT, a.value()-b.value()),
     '*': lambda a,b: Value(Type.INT, a.value()*b.value()),
     '/': lambda a,b: Value(Type.INT, a.value()//b.value()),  # // for integer ops
     '%': lambda a,b: Value(Type.INT, a.value()%b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.STRING] = {
     '+': lambda a,b: Value(Type.STRING, a.value()+b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.BOOL] = {
     '&': lambda a,b: Value(Type.BOOL, a.value() and b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '|': lambda a,b: Value(Type.BOOL, a.value() or b.value())
    }

  def _compute_indentation(self, program):
    self.indents = [len(line) - len(line.lstrip(' ')) for line in program]

  def _find_first_instruction(self, funcname):
    func_info = self.func_manager.get_function_info(funcname)
    if not func_info:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function")

    return func_info.start_ip

  # given a token name (e.g., x, 17, True, "foo"), give us a Value object associated with it
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip)
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == Interpreter.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)

    # look in environments for variable
    val = self.env_manager.get(token)
    if val != None:
      return val

    if "." in token: #check if token is an obj member
          #print("entered get val for obj")
          possible_obj = token.split(".")
          obj_name = possible_obj[0]
          obj_member = possible_obj[1]
          if self.env_manager.get(obj_name) != None: #obj is a var
            obj_value = self._get_value(obj_name)
            #print(obj_value.value())
            obj_value_type = obj_value.type()
            obj_value_value = obj_value.value() #which is a dict of all member vars
            if obj_value_type != Type.OBJECT:
              super().error(ErrorType.TYPE_ERROR,"cannot use dot operator on non_object", self.ip)
            else: #valid obj var
              if obj_member not in obj_value_value: #uknown member var
                super().error(ErrorType.NAME_ERROR,"unknown member variable", self.ip)
              else: #valid member var
                #print(f'obj member value: {obj_value_value[obj_member]}')
                #if token == "l.next":
                  #print(f'value of l.next: {obj_value_value[obj_member].value()}')
                return obj_value_value[obj_member]
    # not found
    super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip)

  # given a variable name and a Value object, associate the name with the value
  def _set_value(self, varname, to_value_type):
    value_type = self.env_manager.get(varname)
    if value_type == None:
      super().error(ErrorType.NAME_ERROR,f"Assignment of unknown variable {varname}", self.ip)
    value_type.set(to_value_type)

  # bind the result[s,i,b] variable in the calling function's scope to the proper Value object
  def _set_result(self, value_type):
    # always stores result in the highest-level block scope for a function, so nested if/while blocks
    # don't each have their own version of result
    result_var = InterpreterBase.RESULT_DEF + self.type_to_result[value_type.type()]
    self.env_manager.create_new_symbol(result_var, True)  # create in top block if it doesn't exist
    if value_type.type() == Type.OBJECT:
      self.env_manager.set(result_var, copy.deepcopy(value_type))
    else:
      self.env_manager.set(result_var, copy.copy(value_type))
    #print(f'result value: {self._get_value(result_var).value()}')

  # evaluate expressions in prefix notation: + 5 * 6 x
  def _eval_expression(self, tokens):
    stack = []
    #print(f'tokens: {tokens}')
    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip)
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip)
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip)
        stack.append(Value(Type.BOOL, not v1.value()))
      else:
        value_type = None
        if "." in token: #check if token is an obj member
          #print("entered obj eval")
          #if token == "l.next":
            #print(f'value of l.next: {self._get_value(token)}')

          #if token == "l.end":
            #print(f'value of l.end: {self._get_value(token).value()}')
          possible_obj = token.split(".")
          obj_name = possible_obj[0]
          obj_member = possible_obj[1]
          if self.env_manager.get(obj_name) != None: #obj is a var
            obj_value = self._get_value(obj_name)
            obj_value_type = obj_value.type()
            obj_value_value = obj_value.value() #which is a dict of all member vars
            if obj_value_type != Type.OBJECT:
              super().error(ErrorType.TYPE_ERROR,"cannot use dot operator on non_object", self.ip)
            else: #valid obj var
              if obj_member not in obj_value_value: #uknown member var
                super().error(ErrorType.NAME_ERROR,"unknown member variable", self.ip)
              else: #valid member var
                #print(f'obj member value in express eval: {obj_value_value[obj_member]}')
                value_type = obj_value_value[obj_member]
                #super().error(ErrorType.NAME_ERROR,"breakpoint", self.ip)
        else:
          value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      #print(stack)
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip)

    return stack[0]



# if __name__ == "__main__":
#     program = ['func main void\n', '  var object o\n', '  assign o.a 100\n', '  if True\n', '    assign o.a "bar"\n', '    var object o\n', '    assign o.b True\n', '    if == o.b False\n', '      funccall print o.a\n', '    else\n', '      assign o.a 10\n', '    endif\n', '    var int y\n', '    assign y - o.a 8\n', '  endif\n', '  var int z\n', '  assign z + o.a 5\n', 'endfunc\n']
#     interpreter = Interpreter()
#     interpreter.run(program)