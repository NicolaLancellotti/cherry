# Cherry Grammar

## Lexical Structure

**Identifiers**  
identifier → identifier-head identifier-characters<sub>opt</sub>  

identifier-head → `Upper or lowercase letter A through Z`  
identifier-characters → identifier-character identifier-characters<sub>opt</sub>      
identifier-character → identifier-head  
identifier-character → `Digit 0 through 9`  

**Literals**  
literal → decimal-literal    
decimal-literal → decimal-digit decimal-digits<sub>opt</sub>    
decimal-digit → `Digit 0 through 9`    
decimal-digits → decimal-digit decimal-digits<sub>opt</sub>     

## Expressions  
expression → function-call-expression  
expression → primary-expression  
expression → variable-expression  
expression → struct-expression  

**Primary expressions**    
primary-expression → literal-expression    

**Literal expressions**  
literal-expression → literal

**Call expression**  
function-call-expression → identifier function-call-argument-clause    
function-call-argument-clause → `(` `)`  
function-call-argument-clause → `(` function-call-argument-list `)`       
function-call-argument-list → function-call-argument    
function-call-argument-list → function-call-argument `,` function-call-argument-list  
function-call-argument → expression    

**Variable expression**
variable-expression → identifier  

**Struct constructor**  
struct-expression → type `{` struct-expression-argument-list `}`         
struct-expression-argument-list → expression    
struct-expression-argument-list → expression `,` function-call-argument-list  

## Statements  
statement → expression `;`  
statements → statement statements<sub>opt</sub>    

## Declarations  
declaration → function-declaration  
declaration → struct-declaration    
declarations → declaration declaration<sub>opt</sub>    

**Top-level declaration**  
top-level-declaration → declarations   

**Function declaration**  
function-declaration → `fun` function-name function-signature  function-body  
function-name → identifier  
function-signature → parameter-clause  
parameter-clause → `(` `)`    
parameter-clause → `(` parameter-list `)`  
parameter-list → parameter  
parameter-list → parameter , parameter-list  
parameter → parameter-name type-annotation  
parameter-name → identifier  
type-annotation → `:` type   
function-body → `{` statements<sub>opt</sub> `}`  

## Types
type → identifier    
struct-declaration → `struct` type `{`  struct-members<sub>opt</sub> `}`  
struct-members → struct-member  
struct-members → struct-member `,` struct-members    
struct-member → identifier type-annotation  