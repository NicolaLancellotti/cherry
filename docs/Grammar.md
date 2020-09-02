# Cherry Grammar

##Lexical Structure

**Grammar of an identifier**  
identifier → identifier-head identifier-characters<sub>opt</sub>  

identifier-head → `Upper or lowercase letter A through Z`  
identifier-characters → identifier-character identifier-characters<sub>opt</sub>      
identifier-character → identifier-head  
identifier-character → `Digit 0 through 9`  

**Literal**  
literal → decimal-literal    
decimal-literal → decimal-digit decimal-digits<sub>opt</sub>    
decimal-digit → `Digit 0 or 1`    
decimal-digits → decimal-digit decimal-digits<sub>opt</sub>     

## Expressions  
expression → function-call-expression  
expression → primary-expression  

**Primary expressions**    
primary-expression → literal-expression    

**Literal expressions**  
literal-expression → literal

**Call expression**  
function-call-expression → identifier function-call-argument-clause    
function-call-argument-clause → `(` `)`  
function-call-argument-clause → `(` function-call-argument-list `)`       
function-call-argument-list → expression  
function-call-argument-list → expression `,` function-call-argument-list      

## Statements  
statement → expression `;`  
statements → statement statements<sub>opt</sub>    

## Declarations  
declaration → function-declaration  
declarations → declaration declaration<sub>opt</sub>    

**Top-level declaration**  
top-level-declaration → declarations   

**Function declaration**  
function-declaration → `fun` function-name function-signature  function-body  
function-name → identifier  
function-signature → parameter-clause  
parameter-clause → `(` `)`    
function-body → `{` statements<sub>opt</sub> `}`  