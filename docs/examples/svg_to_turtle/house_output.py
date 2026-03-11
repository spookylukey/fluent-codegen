# Generated from house.svg by svg_to_turtle.py
# Do not edit — regenerate from the SVG source.
import turtle


def _draw_beam(t: turtle.Turtle) -> None:
    t.penup()
    t.goto(0.0, 0.0)
    t.pendown()
    t.goto(80.0, 0.0)


def draw(t: turtle.Turtle) -> None:
    t.penup()
    t.goto(40.0, 60.0)
    t.pendown()
    t.goto(40.0, 140.0)
    t.penup()
    t.goto(120.0, 60.0)
    t.pendown()
    t.goto(120.0, 140.0)
    t.penup()
    t.goto(30.0, 60.0)
    t.pendown()
    t.goto(80.0, 20.0)
    t.penup()
    t.goto(80.0, 20.0)
    t.pendown()
    t.goto(130.0, 60.0)
    pos = t.position()
    heading = t.heading()
    t.penup()
    t.goto(pos[0] + 40.0, pos[1] + 60.0)
    _draw_beam(t)
    t.penup()
    t.goto(pos)
    t.setheading(heading)
    pos_2 = t.position()
    heading_2 = t.heading()
    t.penup()
    t.goto(pos_2[0] + 40.0, pos_2[1] + 140.0)
    _draw_beam(t)
    t.penup()
    t.goto(pos_2)
    t.setheading(heading_2)


if __name__ == "__main__":
    t = turtle.Turtle()
    draw(t)
    turtle.done()
