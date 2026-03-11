# Generated from house.svg by svg_to_turtle.py
# Do not edit — regenerate from the SVG source.
import turtle


def _draw_beam(t: turtle.Turtle, start: tuple[int, int]) -> None:
    t.penup()
    t.goto(start[0] + 0.0, start[1] + 0.0)
    t.pendown()
    t.goto(start[0] + 80.0, start[1] + 0.0)


def _draw_wall(t: turtle.Turtle, start: tuple[int, int]) -> None:
    t.penup()
    t.goto(start[0] + 0.0, start[1] + 0.0)
    t.pendown()
    t.goto(start[0] + 0.0, start[1] + 80.0)


def draw(t: turtle.Turtle) -> None:
    start = (0, 0)
    pos = t.position()
    heading = t.heading()
    _draw_wall(t, (40, 60))
    t.penup()
    t.goto(pos)
    t.setheading(heading)
    pos_2 = t.position()
    heading_2 = t.heading()
    _draw_wall(t, (120, 60))
    t.penup()
    t.goto(pos_2)
    t.setheading(heading_2)
    t.penup()
    t.goto(start[0] + 30.0, start[1] + 60.0)
    t.pendown()
    t.goto(start[0] + 80.0, start[1] + 20.0)
    t.penup()
    t.goto(start[0] + 80.0, start[1] + 20.0)
    t.pendown()
    t.goto(start[0] + 130.0, start[1] + 60.0)
    pos_3 = t.position()
    heading_3 = t.heading()
    _draw_beam(t, (40, 60))
    t.penup()
    t.goto(pos_3)
    t.setheading(heading_3)
    pos_4 = t.position()
    heading_4 = t.heading()
    _draw_beam(t, (40, 140))
    t.penup()
    t.goto(pos_4)
    t.setheading(heading_4)


if __name__ == "__main__":
    t = turtle.Turtle()
    draw(t)
    turtle.done()
