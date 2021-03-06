# Cherry Grammar  

## Lexical Structure  

**GRAMMAR OF AN IDENTIFIER**  
identifier → identifier-head identifier-characters<sub>opt</sub>  
identifier-head → `Upper or lowercase letter A through Z`  
identifier-characters → identifier-character identifier-characters<sub>opt</sub>  
identifier-character → identifier-head  
identifier-character → `Digit 0 through 9`  

**GRAMMAR OF A LITERAL**  
literal → unit-literal  
literal → boolean-literal  
literal → decimal-literal   

unit-literal  → `(` `)`   
boolean-literal → `true`  
boolean-literal → `false`   
decimal-literal → decimal-digit decimal-digits<sub>opt</sub>    
decimal-digit → `Digit 0 through 9`  
decimal-digits → decimal-digit decimal-digits<sub>opt</sub>    

## Expressions
expression → lvalue  
expression → rvalue   

**GRAMMAR OF A LVALUE EXPRESSION**  
lvalue → variable-expression  
lvalue → struct-access  

**GRAMMAR OF A RVALUE EXPRESSION**  
rvalue → literal-expression  
rvalue → function-call-expression  
rvalue → assign-expression  
rvalue → if-expression    
rvalue → while-expression      
rvalue → binary-expression        

**GRAMMAR OF A LITERAL EXPRESSION**  
literal-expression → literal  

**GRAMMAR OF A CALL EXPRESSION**  
function-call-expression → identifier function-call-argument-clause  
function-call-argument-clause → `(` `)`  
function-call-argument-clause → `(` function-call-argument-list `)`  
function-call-argument-list → function-call-argument  
function-call-argument-list → function-call-argument `,` function-call-argument-list  
function-call-argument → expression  

**GRAMMAR OF A VARIABLE EXPRESSION**  
variable-expression → identifier  

**GRAMMAR OF A STRUCT ACCESS EXPRESSION**   
struct-access → lvalue `.` identifier    
struct-access → struct-access `.` identifier  

**GRAMMAR OF AN ASSIGN EXPRESSION**     
assign-expression → lvalue `=` rvalue  
  
**GRAMMAR OF A IF EXPRESSION**    
if-expression → `if` expression block-expression `else` block-expression

**GRAMMAR OF A WHIle EXPRESSION**    
while-expression → `while` expression block-expression  

**GRAMMAR OF A BLOCK EXPRESSION**    
block-expression → `{` statement-list<sub>opt</sub> expression `}`    

**GRAMMAR OF A BINARY EXPRESSION**      
binary-expression → expression operator expression      
operator → `+`    
operator → `-`     
operator → `*`     
operator → `/`      
operator → `%`  
operator → `and`  
operator → `or`   
operator → `eq`  
operator → `neq`   
operator → `lt`  
operator → `le`    
operator → `gt`  
operator → `ge`  

## Declarations  
declaration → function-declaration  
declaration → struct-declaration  
declarations → declaration declarations<sub>opt</sub>  

**GRAMMAR OF A TOP-LEVEL DECLARATION**  
top-level-declaration → declarations  

**GRAMMAR OF A FUNCTION DECLARATION**  
function-declaration → `fn` function-name function-signature  function-body  
function-name → identifier  
function-signature → `(` parameter-list<sub>opt</sub>  `)` `:` type  
parameter-list → parameter `,`<sub>opt</sub>  
parameter-list → parameter `,` parameter-list  
parameter → parameter-name type-annotation  
parameter-name → identifier  
type-annotation → `:` type  
function-body → block-expression  

**GRAMMAR OF A STRUCT DECLARATION**    
type → identifier  
struct-declaration → `struct` type `{`  struct-members<sub>opt</sub> `}`  
struct-members → struct-member `,`<sub>opt</sub>  
struct-members → struct-member `,` struct-members  
struct-member → identifier type-annotation  

**GRAMMAR OF A VARIABLE DECLARATION**  
var-declaration → `var` variable-expression type-annotation `=` rvalue  

## Statements  
statement-list → statement  
statement-list → statement-list statement  

**GRAMMAR OF A STATEMENT**  
statement = expression `;`  
statement = var-declaration `;`  
