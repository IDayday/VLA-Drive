"""User-selected best measured IL -> unchanged attention+memory Flow GRPO."""
from scripts.analysis.simwam_recipe.train import main
from starVLA.rl.qwen_world_postcot.selected_parent_runtime import SelectedParentMemoryRuntime

if __name__ == '__main__':
    main(SelectedParentMemoryRuntime())
