# Input coloring to prevent MSJs

## Dataset consists of:

* Many-shot jailbreaks  
  * User turn contains an “embedded conversation” with randomly varying human and assistant tags (that differ from the special tokens used in the “true format”). In the embedded conversation, the assistant is shown doing an undesired behavior.  
  * The undesired behavior comes in two variants:  
    * Giving answers to harmful questions  
    * Insulting the user  
  * A “jailbreak” MSJ demonstration consists of the jailbreak attempt in the user turn, followed by an assistant turn that continues the pattern (answers harmfully or insults the user)  
  * A “recovery” MSJ demonstration consists of the attempt in the user turn, followed by an assistant turn that does not continue the pattern but rather instead follows the correct policy (refuses to answer the harmful question or answers without any insults)  
* Single-turn harmful questions  
* Single-turn regular questions  
* Regular conversations  
  * A few user/assistant back-and-forths with regular content that forms a coherent conversation

## Intervention:

* Instead of just using special token delimiters around the “true” human and assistant turns, we also “color” the human and assistant tokens by either modifying the embedding or residual stream (we test both) by adding a constant vector (and optionally projecting out a different constant vector). This can be thought of as a “role embedding” that indicates what role a token belongs to at every token position.  
* We finetune the model with this intervention on:  
  * MSJ \+ recovery  
  * Single-turn harmful questions \+ refusal  
  * Single-turn regular questions \+ regular answers  
  * Regular conversations

## We assess:

* NLL(final assistant response) vs. N shots in MSJ prompt  
  * Both on jailbreaks and recoveries  
  * We want slope of jailbreak response vs. N shots in log-log space to go down (become less steep)  
  * We want absolute NLLs for jailbreak responses to go up (become less likely)  
  * We want absolute NLLs for recovery responses to go down (become more likely)  
* NLL(final assistant response) vs. N shots in regular conversations  
  * We want this to stay roughly the same