??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	?d*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:d*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d? * 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	d? *
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:? *
dtype0
t
value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namevalue/kernel
m
 value/kernel/Read/ReadVariableOpReadVariableOpvalue/kernel*
_output_shapes

:d*
dtype0
l

value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
value/bias
e
value/bias/Read/ReadVariableOpReadVariableOp
value/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
 
8
0
1
2
3
4
5
$6
%7
8
0
1
2
3
4
5
$6
%7
 
?
.layer_regularization_losses
/metrics
		variables

trainable_variables
regularization_losses
0non_trainable_variables
1layer_metrics

2layers
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3layer_regularization_losses
4metrics
	variables
trainable_variables
regularization_losses
5non_trainable_variables
6layer_metrics

7layers
 
 
 
?
8layer_regularization_losses
9metrics
	variables
trainable_variables
regularization_losses
:non_trainable_variables
;layer_metrics

<layers
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
=layer_regularization_losses
>metrics
	variables
trainable_variables
regularization_losses
?non_trainable_variables
@layer_metrics

Alayers
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Blayer_regularization_losses
Cmetrics
 	variables
!trainable_variables
"regularization_losses
Dnon_trainable_variables
Elayer_metrics

Flayers
XV
VARIABLE_VALUEvalue/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
value/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
Glayer_regularization_losses
Hmetrics
&	variables
'trainable_variables
(regularization_losses
Inon_trainable_variables
Jlayer_metrics

Klayers
 
 
 
?
Llayer_regularization_losses
Mmetrics
*	variables
+trainable_variables
,regularization_losses
Nnon_trainable_variables
Olayer_metrics

Players
 
 
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_statePlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_stateconv2d_6/kernelconv2d_6/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasvalue/kernel
value/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????? :?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_154169435
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp value/kernel/Read/ReadVariableOpvalue/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_save_154169696
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasvalue/kernel
value/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference__traced_restore_154169730??
?	
?
G__inference_dense_12_layer_call_and_return_conditional_losses_154169199

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169282	
state
conv2d_6_154169169
conv2d_6_154169171
dense_12_154169210
dense_12_154169212
dense_13_154169236
dense_13_154169238
value_154169275
value_154169277
identity

identity_1?? conv2d_6/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?value/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallstateconv2d_6_154169169conv2d_6_154169171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1541691582"
 conv2d_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_1541691802
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_154169210dense_12_154169212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_1541691992"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_154169236dense_13_154169238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_1541692252"
 dense_13/StatefulPartitionedCall?
policy/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_1541692462
policy/PartitionedCall?
value/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0value_154169275value_154169277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_value_layer_call_and_return_conditional_losses_1541692642
value/StatefulPartitionedCall?
IdentityIdentity&value/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_namestate
?
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169339

inputs
conv2d_6_154169315
conv2d_6_154169317
dense_12_154169321
dense_12_154169323
dense_13_154169326
dense_13_154169328
value_154169332
value_154169334
identity

identity_1?? conv2d_6/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?value/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_154169315conv2d_6_154169317*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1541691582"
 conv2d_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_1541691802
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_154169321dense_12_154169323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_1541691992"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_154169326dense_13_154169328*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_1541692252"
 dense_13/StatefulPartitionedCall?
policy/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_1541692462
policy/PartitionedCall?
value/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0value_154169332value_154169334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_value_layer_call_and_return_conditional_losses_1541692642
value/StatefulPartitionedCall?
IdentityIdentity&value/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_6_layer_call_and_return_conditional_losses_154169180

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169503

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource(
$value_matmul_readvariableop_resource)
%value_biasadd_readvariableop_resource
identity

identity_1??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?value/BiasAdd/ReadVariableOp?value/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_6/Sigmoids
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeconv2d_6/Sigmoid:y:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_12/Sigmoid?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	d? *
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/Sigmoid:y:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_13/BiasAddy
policy/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
policy/Softmax?
value/MatMul/ReadVariableOpReadVariableOp$value_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
value/MatMul/ReadVariableOp?
value/MatMulMatMuldense_12/Sigmoid:y:0#value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value/MatMul?
value/BiasAdd/ReadVariableOpReadVariableOp%value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
value/BiasAdd/ReadVariableOp?
value/BiasAddBiasAddvalue/MatMul:product:0$value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value/BiasAdd?
IdentityIdentityvalue/BiasAdd:output:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/Softmax:softmax:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2<
value/BiasAdd/ReadVariableOpvalue/BiasAdd/ReadVariableOp2:
value/MatMul/ReadVariableOpvalue/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_policy_layer_call_and_return_conditional_losses_154169246

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:?????????? 2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169389

inputs
conv2d_6_154169365
conv2d_6_154169367
dense_12_154169371
dense_12_154169373
dense_13_154169376
dense_13_154169378
value_154169382
value_154169384
identity

identity_1?? conv2d_6/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?value/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_154169365conv2d_6_154169367*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1541691582"
 conv2d_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_1541691802
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_154169371dense_12_154169373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_1541691992"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_154169376dense_13_154169378*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_1541692252"
 dense_13/StatefulPartitionedCall?
policy/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_1541692462
policy/PartitionedCall?
value/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0value_154169382value_154169384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_value_layer_call_and_return_conditional_losses_1541692642
value/StatefulPartitionedCall?
IdentityIdentity&value/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_12_layer_call_and_return_conditional_losses_154169591

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_value_layer_call_and_return_conditional_losses_154169264

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
G__inference_dense_13_layer_call_and_return_conditional_losses_154169610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
+__inference_model_6_layer_call_fn_154169526

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????:?????????? **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_1541693392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_value_layer_call_and_return_conditional_losses_154169629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
d
H__inference_flatten_6_layer_call_and_return_conditional_losses_154169575

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_model_6_layer_call_fn_154169360	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????:?????????? **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_1541693392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_namestate
?
?
"__inference__traced_save_154169696
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop+
'savev2_value_kernel_read_readvariableop)
%savev2_value_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop'savev2_value_kernel_read_readvariableop%savev2_value_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*b
_input_shapesQ
O: :::	?d:d:	d? :? :d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?d: 

_output_shapes
:d:%!

_output_shapes
:	d? :!

_output_shapes	
:? :$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: 
?

?
+__inference_model_6_layer_call_fn_154169410	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????:?????????? **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_1541693892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_namestate
?
?
,__inference_dense_12_layer_call_fn_154169600

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_1541691992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
%__inference__traced_restore_154169730
file_prefix$
 assignvariableop_conv2d_6_kernel$
 assignvariableop_1_conv2d_6_bias&
"assignvariableop_2_dense_12_kernel$
 assignvariableop_3_dense_12_bias&
"assignvariableop_4_dense_13_kernel$
 assignvariableop_5_dense_13_bias#
assignvariableop_6_value_kernel!
assignvariableop_7_value_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_13_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_13_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_value_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_value_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169309	
state
conv2d_6_154169285
conv2d_6_154169287
dense_12_154169291
dense_12_154169293
dense_13_154169296
dense_13_154169298
value_154169302
value_154169304
identity

identity_1?? conv2d_6/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?value/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallstateconv2d_6_154169285conv2d_6_154169287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1541691582"
 conv2d_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_1541691802
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_154169291dense_12_154169293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_1541691992"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_154169296dense_13_154169298*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_1541692252"
 dense_13/StatefulPartitionedCall?
policy/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_1541692462
policy/PartitionedCall?
value/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0value_154169302value_154169304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_value_layer_call_and_return_conditional_losses_1541692642
value/StatefulPartitionedCall?
IdentityIdentity&value/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/PartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_namestate
?
a
E__inference_policy_layer_call_and_return_conditional_losses_154169643

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:?????????? 2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
F
*__inference_policy_layer_call_fn_154169648

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_1541692462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_154169560

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?2
?
$__inference__wrapped_model_154169143	
state3
/model_6_conv2d_6_conv2d_readvariableop_resource4
0model_6_conv2d_6_biasadd_readvariableop_resource3
/model_6_dense_12_matmul_readvariableop_resource4
0model_6_dense_12_biasadd_readvariableop_resource3
/model_6_dense_13_matmul_readvariableop_resource4
0model_6_dense_13_biasadd_readvariableop_resource0
,model_6_value_matmul_readvariableop_resource1
-model_6_value_biasadd_readvariableop_resource
identity

identity_1??'model_6/conv2d_6/BiasAdd/ReadVariableOp?&model_6/conv2d_6/Conv2D/ReadVariableOp?'model_6/dense_12/BiasAdd/ReadVariableOp?&model_6/dense_12/MatMul/ReadVariableOp?'model_6/dense_13/BiasAdd/ReadVariableOp?&model_6/dense_13/MatMul/ReadVariableOp?$model_6/value/BiasAdd/ReadVariableOp?#model_6/value/MatMul/ReadVariableOp?
&model_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_6/conv2d_6/Conv2D/ReadVariableOp?
model_6/conv2d_6/Conv2DConv2Dstate.model_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_6/conv2d_6/Conv2D?
'model_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/conv2d_6/BiasAdd/ReadVariableOp?
model_6/conv2d_6/BiasAddBiasAdd model_6/conv2d_6/Conv2D:output:0/model_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_6/conv2d_6/BiasAdd?
model_6/conv2d_6/SigmoidSigmoid!model_6/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_6/conv2d_6/Sigmoid?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshapemodel_6/conv2d_6/Sigmoid:y:0 model_6/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
model_6/flatten_6/Reshape?
&model_6/dense_12/MatMul/ReadVariableOpReadVariableOp/model_6_dense_12_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02(
&model_6/dense_12/MatMul/ReadVariableOp?
model_6/dense_12/MatMulMatMul"model_6/flatten_6/Reshape:output:0.model_6/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_6/dense_12/MatMul?
'model_6/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'model_6/dense_12/BiasAdd/ReadVariableOp?
model_6/dense_12/BiasAddBiasAdd!model_6/dense_12/MatMul:product:0/model_6/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_6/dense_12/BiasAdd?
model_6/dense_12/SigmoidSigmoid!model_6/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
model_6/dense_12/Sigmoid?
&model_6/dense_13/MatMul/ReadVariableOpReadVariableOp/model_6_dense_13_matmul_readvariableop_resource*
_output_shapes
:	d? *
dtype02(
&model_6/dense_13/MatMul/ReadVariableOp?
model_6/dense_13/MatMulMatMulmodel_6/dense_12/Sigmoid:y:0.model_6/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model_6/dense_13/MatMul?
'model_6/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02)
'model_6/dense_13/BiasAdd/ReadVariableOp?
model_6/dense_13/BiasAddBiasAdd!model_6/dense_13/MatMul:product:0/model_6/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model_6/dense_13/BiasAdd?
model_6/policy/SoftmaxSoftmax!model_6/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
model_6/policy/Softmax?
#model_6/value/MatMul/ReadVariableOpReadVariableOp,model_6_value_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02%
#model_6/value/MatMul/ReadVariableOp?
model_6/value/MatMulMatMulmodel_6/dense_12/Sigmoid:y:0+model_6/value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/value/MatMul?
$model_6/value/BiasAdd/ReadVariableOpReadVariableOp-model_6_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model_6/value/BiasAdd/ReadVariableOp?
model_6/value/BiasAddBiasAddmodel_6/value/MatMul:product:0,model_6/value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/value/BiasAdd?
IdentityIdentity model_6/policy/Softmax:softmax:0(^model_6/conv2d_6/BiasAdd/ReadVariableOp'^model_6/conv2d_6/Conv2D/ReadVariableOp(^model_6/dense_12/BiasAdd/ReadVariableOp'^model_6/dense_12/MatMul/ReadVariableOp(^model_6/dense_13/BiasAdd/ReadVariableOp'^model_6/dense_13/MatMul/ReadVariableOp%^model_6/value/BiasAdd/ReadVariableOp$^model_6/value/MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity?

Identity_1Identitymodel_6/value/BiasAdd:output:0(^model_6/conv2d_6/BiasAdd/ReadVariableOp'^model_6/conv2d_6/Conv2D/ReadVariableOp(^model_6/dense_12/BiasAdd/ReadVariableOp'^model_6/dense_12/MatMul/ReadVariableOp(^model_6/dense_13/BiasAdd/ReadVariableOp'^model_6/dense_13/MatMul/ReadVariableOp%^model_6/value/BiasAdd/ReadVariableOp$^model_6/value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2R
'model_6/conv2d_6/BiasAdd/ReadVariableOp'model_6/conv2d_6/BiasAdd/ReadVariableOp2P
&model_6/conv2d_6/Conv2D/ReadVariableOp&model_6/conv2d_6/Conv2D/ReadVariableOp2R
'model_6/dense_12/BiasAdd/ReadVariableOp'model_6/dense_12/BiasAdd/ReadVariableOp2P
&model_6/dense_12/MatMul/ReadVariableOp&model_6/dense_12/MatMul/ReadVariableOp2R
'model_6/dense_13/BiasAdd/ReadVariableOp'model_6/dense_13/BiasAdd/ReadVariableOp2P
&model_6/dense_13/MatMul/ReadVariableOp&model_6/dense_13/MatMul/ReadVariableOp2L
$model_6/value/BiasAdd/ReadVariableOp$model_6/value/BiasAdd/ReadVariableOp2J
#model_6/value/MatMul/ReadVariableOp#model_6/value/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????

_user_specified_namestate
?+
?
F__inference_model_6_layer_call_and_return_conditional_losses_154169469

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource(
$value_matmul_readvariableop_resource)
%value_biasadd_readvariableop_resource
identity

identity_1??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?value/BiasAdd/ReadVariableOp?value/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_6/Sigmoids
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeconv2d_6/Sigmoid:y:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_12/Sigmoid?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	d? *
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/Sigmoid:y:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_13/BiasAddy
policy/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
policy/Softmax?
value/MatMul/ReadVariableOpReadVariableOp$value_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
value/MatMul/ReadVariableOp?
value/MatMulMatMuldense_12/Sigmoid:y:0#value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value/MatMul?
value/BiasAdd/ReadVariableOpReadVariableOp%value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
value/BiasAdd/ReadVariableOp?
value/BiasAddBiasAddvalue/MatMul:product:0$value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value/BiasAdd?
IdentityIdentityvalue/BiasAdd:output:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypolicy/Softmax:softmax:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2<
value/BiasAdd/ReadVariableOpvalue/BiasAdd/ReadVariableOp2:
value/MatMul/ReadVariableOpvalue/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_154169158

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_value_layer_call_fn_154169638

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_value_layer_call_and_return_conditional_losses_1541692642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
,__inference_conv2d_6_layer_call_fn_154169569

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1541691582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_13_layer_call_fn_154169619

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_1541692252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
G__inference_dense_13_layer_call_and_return_conditional_losses_154169225

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
I
-__inference_flatten_6_layer_call_fn_154169580

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_1541691802
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_model_6_layer_call_fn_154169549

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????:?????????? **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_1541693892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_signature_wrapper_154169435	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':?????????? :?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__wrapped_model_1541691432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_namestate"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
state6
serving_default_state:0?????????;
policy1
StatefulPartitionedCall:0?????????? 9
value0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?8
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses"?5
_tf_keras_network?5{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}, "name": "state", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["state", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 100, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "policy", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "policy", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}], "input_layers": [["state", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["value", 0, 0], ["policy", 0, 0]]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}, "name": "state", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["state", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 100, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "policy", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "policy", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}], "input_layers": [["state", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["value", 0, 0], ["policy", 0, 0]]}}}, "training_config": {"loss": "mse", "metrics": ["mae"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.1, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 8]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 100, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 288]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
\__call__
*]&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "policy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "policy", "trainable": true, "dtype": "float32", "activation": "softmax"}}
"
	optimizer
X
0
1
2
3
4
5
$6
%7"
trackable_list_wrapper
X
0
1
2
3
4
5
$6
%7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.layer_regularization_losses
/metrics
		variables

trainable_variables
regularization_losses
0non_trainable_variables
1layer_metrics

2layers
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
`serving_default"
signature_map
):'2conv2d_6/kernel
:2conv2d_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3layer_regularization_losses
4metrics
	variables
trainable_variables
regularization_losses
5non_trainable_variables
6layer_metrics

7layers
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8layer_regularization_losses
9metrics
	variables
trainable_variables
regularization_losses
:non_trainable_variables
;layer_metrics

<layers
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
": 	?d2dense_12/kernel
:d2dense_12/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=layer_regularization_losses
>metrics
	variables
trainable_variables
regularization_losses
?non_trainable_variables
@layer_metrics

Alayers
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
": 	d? 2dense_13/kernel
:? 2dense_13/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Blayer_regularization_losses
Cmetrics
 	variables
!trainable_variables
"regularization_losses
Dnon_trainable_variables
Elayer_metrics

Flayers
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:d2value/kernel
:2
value/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Glayer_regularization_losses
Hmetrics
&	variables
'trainable_variables
(regularization_losses
Inon_trainable_variables
Jlayer_metrics

Klayers
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Llayer_regularization_losses
Mmetrics
*	variables
+trainable_variables
,regularization_losses
Nnon_trainable_variables
Olayer_metrics

Players
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
+__inference_model_6_layer_call_fn_154169526
+__inference_model_6_layer_call_fn_154169360
+__inference_model_6_layer_call_fn_154169410
+__inference_model_6_layer_call_fn_154169549?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_154169143?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
state?????????
?2?
F__inference_model_6_layer_call_and_return_conditional_losses_154169503
F__inference_model_6_layer_call_and_return_conditional_losses_154169282
F__inference_model_6_layer_call_and_return_conditional_losses_154169469
F__inference_model_6_layer_call_and_return_conditional_losses_154169309?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_6_layer_call_fn_154169569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_154169560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_6_layer_call_fn_154169580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_6_layer_call_and_return_conditional_losses_154169575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_12_layer_call_fn_154169600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_12_layer_call_and_return_conditional_losses_154169591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_13_layer_call_fn_154169619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_13_layer_call_and_return_conditional_losses_154169610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_value_layer_call_fn_154169638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_value_layer_call_and_return_conditional_losses_154169629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_policy_layer_call_fn_154169648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_policy_layer_call_and_return_conditional_losses_154169643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_154169435state"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_154169143?$%6?3
,?)
'?$
state?????????
? "Z?W
+
policy!?
policy?????????? 
(
value?
value??????????
G__inference_conv2d_6_layer_call_and_return_conditional_losses_154169560l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_6_layer_call_fn_154169569_7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_dense_12_layer_call_and_return_conditional_losses_154169591]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
,__inference_dense_12_layer_call_fn_154169600P0?-
&?#
!?
inputs??????????
? "??????????d?
G__inference_dense_13_layer_call_and_return_conditional_losses_154169610]/?,
%?"
 ?
inputs?????????d
? "&?#
?
0?????????? 
? ?
,__inference_dense_13_layer_call_fn_154169619P/?,
%?"
 ?
inputs?????????d
? "??????????? ?
H__inference_flatten_6_layer_call_and_return_conditional_losses_154169575a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
-__inference_flatten_6_layer_call_fn_154169580T7?4
-?*
(?%
inputs?????????
? "????????????
F__inference_model_6_layer_call_and_return_conditional_losses_154169282?$%>?;
4?1
'?$
state?????????
p

 
? "L?I
B??
?
0/0?????????
?
0/1?????????? 
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_154169309?$%>?;
4?1
'?$
state?????????
p 

 
? "L?I
B??
?
0/0?????????
?
0/1?????????? 
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_154169469?$%??<
5?2
(?%
inputs?????????
p

 
? "L?I
B??
?
0/0?????????
?
0/1?????????? 
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_154169503?$%??<
5?2
(?%
inputs?????????
p 

 
? "L?I
B??
?
0/0?????????
?
0/1?????????? 
? ?
+__inference_model_6_layer_call_fn_154169360?$%>?;
4?1
'?$
state?????????
p

 
? ">?;
?
0?????????
?
1?????????? ?
+__inference_model_6_layer_call_fn_154169410?$%>?;
4?1
'?$
state?????????
p 

 
? ">?;
?
0?????????
?
1?????????? ?
+__inference_model_6_layer_call_fn_154169526?$%??<
5?2
(?%
inputs?????????
p

 
? ">?;
?
0?????????
?
1?????????? ?
+__inference_model_6_layer_call_fn_154169549?$%??<
5?2
(?%
inputs?????????
p 

 
? ">?;
?
0?????????
?
1?????????? ?
E__inference_policy_layer_call_and_return_conditional_losses_154169643Z0?-
&?#
!?
inputs?????????? 
? "&?#
?
0?????????? 
? {
*__inference_policy_layer_call_fn_154169648M0?-
&?#
!?
inputs?????????? 
? "??????????? ?
'__inference_signature_wrapper_154169435?$%??<
? 
5?2
0
state'?$
state?????????"Z?W
+
policy!?
policy?????????? 
(
value?
value??????????
D__inference_value_layer_call_and_return_conditional_losses_154169629\$%/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? |
)__inference_value_layer_call_fn_154169638O$%/?,
%?"
 ?
inputs?????????d
? "??????????