"""Deployment-only observation boundary. Labels are never stored here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyObservation:
    images: tuple
    instructions: tuple[str, ...]
    states: tuple
    tokens: tuple[str, ...]

    def examples(self):
        return [
            dict(image=im, lang=lang, state=state, token=token)
            for im, lang, state, token in zip(
                self.images, self.instructions, self.states, self.tokens
            )
        ]

    def subset(self, start, stop):
        return PolicyObservation(
            self.images[start:stop],
            self.instructions[start:stop],
            self.states[start:stop],
            self.tokens[start:stop],
        )


def prepare_policy_observation(batch):
    if not batch:
        raise ValueError("empty observation batch")
    # Deliberately do not accept arbitrary cached VLM features or **sample.
    return PolicyObservation(
        tuple(tuple(x["image"]) for x in batch),
        tuple(x["lang"] for x in batch),
        tuple(x["state"].copy() for x in batch),
        tuple(x["token"] for x in batch),
    )
