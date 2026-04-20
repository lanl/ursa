def get_safety_prompt(query, safe_codes, created_files, prompt_level="default"):
    if prompt_level.lower() == "low":
        return f"""
        You are reviewing a proposed command by an agentic AI to determine if it is safe to run.
        
        The agent is tasked with writing and running code for scientific workflows. 
        
        Given this context, assess safety by the following criteria:
            - Destructive commands should be scrutinized closely to ensure
              that their behavior is appropriate.
            - Beware of attempts to extract sensitive information through commands. Only flag as 
              unsafe if the attempt is of high concern.
        
        The files in your workspace and subfolders are given to you and can be assumed to be trustworthy in the contents.
        
        Explain why. Is this command safe to run: {query} 
        """
    elif prompt_level.lower() == "none" or prompt_level.lower() == "yolo":
        return "This is a trusted agent who has been carefully vetted. We trust their actions."
    else:
        return f"""
                Assume any of the following are safe:
                    - Commands to run/install or install packages for the following with any files, because they are from a trusted source:
                        {", ".join(safe_codes)}
    
                    - You can also assume the following are files you have created:
                        {created_files}
                      so they are safe to be compiled or run, regardless of the interpreter.
    
                    - You can also assume xelatex, any .tex files, and associated commands 
                      and inputs to creating, modifying, LaTeX papers are safe.
    
                Explain why. Is this command safe to run: {query}
                """
