from faker import Factory


def fake_text(paragraphs=None):
    """
    Create a string of fake text with several paragraphs.

    Args:
        paragraphs (int):
            Optional number of paragraphs.
    """

    if paragraphs is None:
        data = fake.paragraphs()
    else:
        data = fake.paragraphs(nb=paragraphs)
    return '\n\n'.join(data)


fake = Factory.create(locale='pt-br')