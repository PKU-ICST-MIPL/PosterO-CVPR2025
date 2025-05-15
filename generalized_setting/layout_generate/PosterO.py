import torch
import numpy as np
import os

class PosterO:
    def __init__(self, llm, layout_planter, sampler):      
        self.llm = llm
        
        self.layout_planter = layout_planter
        self.sampler = sampler
        self.db_size = len(self.layout_planter.db_train)
        
    def debug_from_planter(self, llm_sampl, prompt_dict, label_info, label_rback, N, checkpoint):
        if 'db_valid' in dir(self.layout_planter):
            split = self.layout_planter.db_valid[:checkpoint['valid']+1]
            print(f"Start debuging `annotated test split` with {len(split)}.")
            self.inference_dataset_split(split, llm_sampl, prompt_dict, label_info, label_rback, N, debug=True, split_name="valid", checkpoint=checkpoint['valid'])
        
        if 'db_test' in dir(self.layout_planter):
            split = self.layout_planter.db_test[:checkpoint['test']+1]
            print(f"Start debuging `unannotated test split` with {len(split)}.")
            self.inference_dataset_split(split, llm_sampl, prompt_dict, label_info, label_rback, N, debug=True, split_name="test", checkpoint=checkpoint['test'])

    def inference_from_planter(self, llm_sampl, prompt_dict, label_info, save_path, label_rback, N, checkpoint):
        if 'db_valid' in dir(self.layout_planter):
            split = self.layout_planter.db_valid
            print(f"Start inferencing `annotated test split` with {len(split)}.")
            self.inference_dataset_split(split, llm_sampl, prompt_dict, label_info, label_rback, N, split_name="valid", save_path=save_path, checkpoint=checkpoint['valid'])
        
        if 'db_test' in dir(self.layout_planter):
            split = self.layout_planter.db_test
            print(f"Start inferencing `unannotated test split` with {len(split)}.")
            self.inference_dataset_split(split, llm_sampl, prompt_dict, label_info, label_rback, N, split_name="test", save_path=save_path, checkpoint=checkpoint['test'])

        self.save_inference_result(save_path)
        
    def inference_dataset_split(self, split, llm_sampl, prompt_dict, label_info, label_rback, N, debug=False, split_name=None, save_path=None, checkpoint=0):
        for i in range(checkpoint, len(split)):
            labels = self.layout_planter.db_train[np.random.randint(self.db_size)]['layout']['cls_elem']
            rag_kwargs = {'instance': split[i], 'labels': labels, 'split_name': split_name, 'self_i': i}
            rag_results = self.sampler(**rag_kwargs)
            
            if label_rback:
                if debug:
                    print(f"Old Labels: {labels}")
                # labels = rag_results['label'][0] # almost trivial when rag by random chosen label, non-trivial when rag by instance
                labels = self.layout_planter.intepreter(rag_results['layout_description'][0][1])['cls_elem'] # almost trivial when rag by random chosen label, non-trivial when rag by instance
            else:
                if debug:
                    print(f"Old Labels: {labels}")
                labels = self.layout_planter.intepreter(self.layout_planter.db_train[i]['layout_description'][1])['cls_elem']
            
            rag = "\n".join([prompt_dict['rag_opening'].format(j) + head + svg for j, (head, svg) in enumerate(rag_results['layout_description'])])
            if debug:
                print(f"Instance:\n{split[i]}")
                print(f"Labels: {labels}")
                print(f"RAG:\n{rag}")
                print(f"Prompt appendix: {prompt_dict['pulse_appendix']}")
                
            pulse = self.layout_planter.getSVGPrompt(labels, label_info, split[i]) + prompt_dict['pulse_appendix']
            prompt = "\n".join([prompt_dict['opening'], rag, prompt_dict['rule'], pulse])
            if debug:
                print(f"Pulse: {pulse}")
            
            svg_results = []
            interpret_results = []
            
            while True:
                outputs = self.llm.generate(prompt, sampling_params=llm_sampl)[0].outputs
            
                for output in outputs:
                    try:
                        interpret = self.layout_planter.intepreter(output.text)
                    except:
                        continue
                    
                    svg_results.append(output.text)
                    interpret_results.append(interpret)
                
                if len(interpret_results) >= N:
                    break

            split[i]["generated"] = {
                "svg": svg_results,
                "layout": interpret_results
            }
            
            if debug:
                print(f"Generated SVGs: {svg_results}")
                print(f"Generated Layouts: {interpret_results}")
            
            if save_path is not None:
                self.save_inference_result(save_path, finished=False, checkpoint=i)

    def save_inference_result(self, save_path, finished=True, checkpoint=None):
        if 'db_valid' not in dir(self.layout_planter):
            results = {
                "valid": None,
                "test": self.layout_planter.db_test,
            }
        else:
            results = {
                "valid": self.layout_planter.db_valid,
                "test": self.layout_planter.db_test
            }
        if not finished:
            results["checkpoint"] = checkpoint
            torch.save(results, save_path + ".tmp")
        else:
            torch.save(results, save_path)
            os.remove(save_path + ".tmp")