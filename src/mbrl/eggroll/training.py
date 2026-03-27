"""Shared EGGROLL training utilities.

Provides state management, iterinfo generation, and the common
convert_fitnesses + do_updates + sigma scheduling step shared by
both world model training (mbrl.world_models.eggroll) and
policy search (mbrl.policy_optimizers.eggroll).
"""
